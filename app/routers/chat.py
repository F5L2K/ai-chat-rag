"""对话接口：流式问答、历史、会话管理。

注意：流式接口一旦开始推送就没法回头改状态码，
所以「流开始前的错误」用 HTTPException（有状态码），
「流开始后的错误」直接在流里塞一段错误文字（此时状态码已经是 200）。
"""

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

    # RAG：只有最新这句问题用来检索（检索是找「跟此刻问题相关的资料」）
    system_with_rag = build_rag_prompt(req.message, request_id)
    full_messages = [{"role": "system", "content": system_with_rag}] + history

    def generate():
        """生成器：yield 一次 = 推一块给前端，前端收到一块显示一块。"""
        full_reply = ""
        started = time.perf_counter()
        try:
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=full_messages,
                stream=True,
            )
            for chunk in stream:
                # 每个 chunk 里可能有 delta.content（增量的几个字），也可能没有
                delta = chunk.choices[0].delta
                if delta.content:
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
