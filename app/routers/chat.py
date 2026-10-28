"""对话接口：流式问答、历史、会话管理。

注意：流式接口一旦开始推送就没法回头改状态码，
所以「流开始前的错误」用 HTTPException（有状态码），
「流开始后的错误」直接在流里塞一段错误文字（此时状态码已经是 200）。

引用标记机制：流正常结束后，在 save_message 之后 yield 一段 JSON 标记，
前端检测到标记后解析引用列表并展示。标记不进 full_reply，不落库，
因此数据库里存的始终是干净的模型回复文本。
"""

import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from app.core.config import CHAT_MODEL
from app.core.logging import logger, preview
from app.models.schemas import ChatRequest
from app.services.chat_history import (
    delete_session,
    list_sessions,
    load_history,
    save_message,
)
from app.services.llm import client
from app.services.rag import build_rag_prompt

# 前端靠这个标记识别引用 JSON 段；选足够特殊的字符串避免与正常文本冲突
CITATION_MARKER = "__CITATIONS_JSON__"
# 与引用标记同机制：流末尾在引用 JSON 之后再追加一段检索链路快照，
# 前端解析后展示「🔍 本次检索过程」面板。标记不进 full_reply、不落库。
TRACE_MARKER = "__RETRIEVAL_TRACE_JSON__"
# 推理模型（deepseek-v4-pro 等）会先吐 reasoning_content（思考链）再吐 content（正式回答）。
# 用这对标记把思考过程包起来推给前端：前端折叠展示「💭 思考过程」，
# 不进 full_reply、不落库。非推理模型不吐 reasoning_content，这对标记不会出现，行为与以前一致。
REASONING_START = "__REASONING_START__"
REASONING_END = "__REASONING_END__"

router = APIRouter()


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    # ===== 流开始前：读历史 + 存用户消息 =====
    request_id = getattr(request.state, "request_id", "-")
    history = load_history(req.session_id)
    save_message(req.session_id, "user", req.message)
    history.append({"role": "user", "content": req.message})
    logger.info(
        "[%s] 收到提问 session=%s 历史消息=%d 条 问题=%s",
        request_id, req.session_id, len(history), preview(req.message),
    )

    # RAG：检索 + 拼 prompt，同时拿到引用列表与检索链路快照（流末尾推给前端）
    t_rag_start = time.perf_counter()
    system_with_rag, citations, trace = build_rag_prompt(req.message, request_id)
    t_rag_cost = (time.perf_counter() - t_rag_start) * 1000
    full_messages = [{"role": "system", "content": system_with_rag}] + history
    logger.info("[%s] RAG 检索+拼 prompt 耗时=%.0fms", request_id, t_rag_cost)

    def generate():
        """生成器：yield 一次 = 推一块给前端，前端收到一块显示一块。"""
        full_reply = ""
        reasoning_text = ""          # 思考链原文，不入库
        reasoning_started = False   # 是否已推过 REASONING_START
        reasoning_closed = False     # 是否已推过 REASONING_END
        t_first_chunk = None
        t_first_content = None
        started = time.perf_counter()
        try:
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=full_messages,
                stream=True,
                # DeepSeek 扩展参数：low = 压缩思考链，首字更快；
                # 想恢复深度思考改回 "high" 或删掉这行。
                extra_body={"effort": "low"},
            )
            for chunk in stream:
                if t_first_chunk is None:
                    t_first_chunk = (time.perf_counter() - started) * 1000
                    logger.info("[%s] LLM 首块到达=%.0fms", request_id, t_first_chunk)
                # 每个 chunk 里可能有 delta.content（增量的几个字），也可能没有
                delta = chunk.choices[0].delta
                # 推理模型：先吐 reasoning_content（思考链），再吐 content（正式回答）。
                # 用 getattr 兼容非推理模型的 Delta 对象（没有这个字段）。
                reasoning_chunk = getattr(delta, "reasoning_content", None)
                if reasoning_chunk:
                    if not reasoning_started:
                        yield REASONING_START
                        reasoning_started = True
                    reasoning_text += reasoning_chunk
                    # 思考块原样推给前端，由前端在 START/END 之间收集、折叠展示
                    yield reasoning_chunk
                if delta.content:
                    if t_first_content is None:
                        t_first_content = (time.perf_counter() - started) * 1000
                        logger.info("[%s] LLM 首条正式回答到达=%.0fms（思考占=%.0fms）", request_id, t_first_content, t_first_content - (t_first_chunk or 0))
                    # 思考阶段结束 → 补一个 END 标记，再开始推正式回答
                    if reasoning_started and not reasoning_closed:
                        yield REASONING_END
                        reasoning_closed = True
                    full_reply += delta.content
                    yield delta.content
        except APITimeoutError:
            logger.warning("[%s] 模型响应超时", request_id)
            yield "\n\n[错误] 模型响应超时，请稍后重试"
            return
        except APIConnectionError:
            logger.warning("[%s] 连不上 DeepSeek 服务", request_id)
            yield "\n\n[错误] 连不上 DeepSeek 服务，请稍后重试"
            return
        except RateLimitError:
            logger.warning("[%s] 触发 DeepSeek 限流", request_id)
            yield "\n\n[错误] 请求太频繁，请稍等几秒再试"
            return
        except AuthenticationError as e:
            # 只记异常本身，不打堆栈：避免把请求头里的凭证带进日志
            logger.error("[%s] DeepSeek 鉴权失败，请检查密钥是否有效：%s", request_id, e)
            yield "\n\n[错误] 服务配置有误，请联系管理员"
            return
        except APIStatusError as e:
            logger.error("[%s] DeepSeek 返回异常状态码 status=%s", request_id, e.status_code)
            yield "\n\n[错误] DeepSeek 服务返回错误，请稍后重试"
            return
        except Exception:
            # 兜底：流中间的任何意外
            logger.exception("[%s] 生成过程中出现未预期异常", request_id)
            yield "\n\n[错误] 生成过程中出现异常，请稍后重试"
            return

        # 流正常结束：把完整回复落库
        save_message(req.session_id, "assistant", full_reply)
        logger.info(
            "[%s] 生成完成 回复长度=%d 耗时=%.0fms",
            request_id, len(full_reply), (time.perf_counter() - started) * 1000,
        )

        # 引用标记在落库之后 yield：不进 full_reply、不落库，
        # 前端检测到标记后解析 JSON 并展示引用来源
        if citations:
            yield f"{CITATION_MARKER}{json.dumps(citations, ensure_ascii=False)}"
        # 引用之后再追加检索链路快照（trace 在前、cit 在后时 JSON 会粘连，
        # 固定 trace 在最后，前端按 TRACE_MARKER 切开即可）
        if trace:
            yield f"{TRACE_MARKER}{json.dumps(trace, ensure_ascii=False)}"

    # text/event-stream 是 SSE 标准 MIME，浏览器/客户端知道这是流式响应
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    # system 人设从未入库（它是配置不是数据），查出来的天然就是可见消息
    return {"session_id": session_id, "messages": load_history(session_id)}


@router.get("/sessions")
async def list_all_sessions():
    return {"sessions": list_sessions()}


@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    deleted = delete_session(session_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail={
            "code": "session_not_found",
            "message": "会话不存在",
        })
    return {"deleted": deleted}
