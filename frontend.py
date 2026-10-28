import json
import uuid

import requests
import streamlit as st

# 与后端 chat.py 的 CITATION_MARKER 保持一致
CITATION_MARKER = "__CITATIONS_JSON__"
# 流末尾在引用 JSON 之后追加的检索链路快照标记
TRACE_MARKER = "__RETRIEVAL_TRACE_JSON__"
# 思考过程标记：推理模型先吐思考链再吐正式回答，这对标记把思考过程包起来
REASONING_START = "__REASONING_START__"
REASONING_END = "__REASONING_END__"

BACKEND_CHAT_URL = "http://127.0.0.1:8000/api/chat"
BACKEND_HISTORY_URL = "http://127.0.0.1:8000/api/history"
BACKEND_SESSIONS_URL = "http://127.0.0.1:8000/api/sessions"
BACKEND_KNOWLEDGE_URL = "http://127.0.0.1:8000/api/knowledge"


# ============== 全局样式：窄阅读区 + 消息卡片/气泡 ==============
# 纯 CSS 覆盖，不改任何业务逻辑：
#   1. 主区限宽 900px —— 现在满屏宽，长行阅读很累
#   2. 每条消息做成圆角卡片；用户消息浅蓝底（新版 Streamlit 带
#      data-chat-message-author 属性，老版本命中不了就退化为统一卡片，不报错）
st.markdown(
    """
<style>
section.main .block-container {
    max-width: 900px;
    padding-top: 1.5rem;
}
div[data-testid="stChatMessage"] {
    border: 1px solid #eceef2;
    border-radius: 14px;
    padding: 0.55rem 0.9rem;
    margin-bottom: 0.6rem;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
div[data-chat-message-author="user"] {
    background: #eef4ff;
    border-color: #dbe7ff;
}
</style>
""",
    unsafe_allow_html=True,
)

# 空会话欢迎页上的示例问题：都是当前知识库里有标准答案的题，点一下直接发
EXAMPLE_QUESTIONS = [
    "死锁和饥饿有什么区别？",
    "数据结构里的时间复杂度是怎么算的？",
    "MySQL 里 VARCHAR 和 CHAR 有什么区别？",
]


def backend_error_message(res) -> str:
    # 从错误响应里提取"给人看的那句话"；detail 可能是 dict、字符串或其他形状
    try:
        detail = res.json().get("detail", {})
    except ValueError:
        detail = {}
    if isinstance(detail, dict):
        return detail.get("message", "未知错误")
    return str(detail) or "未知错误"


def stream_ai_reply(session_id: str, message: str):
    """生成器：逐块从后端接收 AI 回复，供 st.write_stream 消费。

    协议（流可能含三种标记，按出现顺序）：
      REASONING_START ... REASONING_END   思考链（推理模型）：收集到 reasoning_buffer，不渲染
      CITATION_MARKER {json}              引用来源：收集到 citations
      TRACE_MARKER {json}                  检索链路快照：收集到 stream_ai_reply.trace

    普通文本块正常 yield 给打字机；思考链和标记 JSON 都不进打字机。
    """
    res = requests.post(
        BACKEND_CHAT_URL,
        json={"session_id": session_id, "message": message},
        stream=True,
        timeout=180,  # 推理模型思考阶段可能较长，给足
    )
    if res.status_code != 200:
        raise RuntimeError(f"出错了（{res.status_code}）：{backend_error_message(res)}")

    # 函数属性在上次调用后可能残留，每次开始先清空
    stream_ai_reply.citations = []
    stream_ai_reply.trace = None
    stream_ai_reply.reasoning = ""

    buffer = ""
    reasoning_buffer = ""
    citation_buffer = ""
    mode = "normal"  # normal | reasoning | citations
    # 标记可能被 chunk 边界截断，保留最长标记长度的字符做滑动窗口
    max_marker_len = max(len(CITATION_MARKER), len(REASONING_START), len(REASONING_END))

    for chunk in res.iter_content(decode_unicode=True):
        if not chunk:
            continue
        buffer += chunk

        # 状态机：反复扫描 buffer，直到没有可识别的标记边界
        while True:
            if mode == "citations":
                # citations 模式下后续全是 JSON（含 trace 标记），整体攒着
                citation_buffer += buffer
                buffer = ""
                break

            if mode == "reasoning":
                end_idx = buffer.find(REASONING_END)
                if end_idx >= 0:
                    reasoning_buffer += buffer[:end_idx]
                    buffer = buffer[end_idx + len(REASONING_END):]
                    mode = "normal"
                else:
                    # END 可能被截断：保留末尾 max_marker_len 个字符，其余落盘
                    if len(buffer) > max_marker_len:
                        reasoning_buffer += buffer[:-max_marker_len]
                        buffer = buffer[-max_marker_len:]
                    break

            # normal 模式：先看 citation 还是 reasoning 先出现
            cit_idx = buffer.find(CITATION_MARKER)
            rea_idx = buffer.find(REASONING_START)
            if cit_idx >= 0 and (rea_idx < 0 or cit_idx < rea_idx):
                if cit_idx > 0:
                    yield buffer[:cit_idx]
                buffer = buffer[cit_idx + len(CITATION_MARKER):]
                mode = "citations"
            elif rea_idx >= 0:
                if rea_idx > 0:
                    yield buffer[:rea_idx]
                buffer = buffer[rea_idx + len(REASONING_START):]
                mode = "reasoning"
            else:
                # 没有任何标记：保留末尾 max_marker_len 个字符防截断
                if len(buffer) > max_marker_len:
                    yield buffer[:-max_marker_len]
                    buffer = buffer[-max_marker_len:]
                break

    # 流结束后收尾
    if mode == "reasoning" and buffer:
        reasoning_buffer += buffer
    if mode == "normal" and buffer:
        yield buffer

    # citation_buffer 里 = 引用 JSON + [TRACE_MARKER + 检索链路 JSON]
    if citation_buffer:
        raw = citation_buffer.strip()
        if TRACE_MARKER in raw:
            cit_part, _, trace_part = raw.partition(TRACE_MARKER)
            try:
                citations = json.loads(cit_part.strip())
            except json.JSONDecodeError:
                citations = []
            try:
                stream_ai_reply.trace = json.loads(trace_part.strip())
            except json.JSONDecodeError:
                stream_ai_reply.trace = None
        else:
            try:
                citations = json.loads(raw)
            except json.JSONDecodeError:
                citations = []
    else:
        citations = []

    stream_ai_reply.reasoning = reasoning_buffer
    stream_ai_reply.citations = citations


# ============== 侧边栏：会话列表 ==============

# 当前会话 ID：优先从 URL 参数读取，刷新页面也不丢；没有则生成新的并写回 URL
if "session_id" not in st.session_state:
    url_sid = st.query_params.get("session_id")
    if url_sid:
        st.session_state.session_id = url_sid
    else:
        st.session_state.session_id = str(uuid.uuid4())
        st.query_params.session_id = st.session_state.session_id


def fetch_sessions() -> list:
    try:
        res = requests.get(BACKEND_SESSIONS_URL, timeout=5)
        if res.status_code == 200:
            return res.json()["sessions"]
    except (requests.ConnectionError, requests.Timeout):
        pass
    return []


def switch_session(session_id: str):
    # 切会话：换钥匙 + 清空本地消息缓存（下次重跑时会重新从后端拉）
    st.session_state.session_id = session_id
    st.session_state.messages = None  # 用 None 标记"需要重新加载"
    st.query_params.session_id = session_id  # 同步到 URL，刷新后还能定位到这个会话


def new_session():
    # 新会话：生成新钥匙 + 清缓存；真正落库等用户发第一句话时由后端完成
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.query_params.session_id = st.session_state.session_id  # 同步到 URL


def delete_session(session_id: str):
    try:
        requests.delete(f"{BACKEND_SESSIONS_URL}/{session_id}", timeout=5)
    except (requests.ConnectionError, requests.Timeout):
        st.error("删除失败：后端没响应")
        return
    # 删的是当前会话 → 自动开一个新的
    if session_id == st.session_state.session_id:
        new_session()
    # 否则保持当前会话不变


# ============== 知识库：上传 / 列表 / 删除 ==============

def upload_knowledge_file(uploaded) -> dict | None:
    """把上传的文件发给后端解析入库。失败时在界面上报错并返回 None。"""
    try:
        res = requests.post(
            f"{BACKEND_KNOWLEDGE_URL}/upload-file",
            files={"file": (uploaded.name, uploaded.getvalue())},
            timeout=300,  # 首次上传要加载 embedding 模型，给足时间
        )
    except (requests.ConnectionError, requests.Timeout):
        st.error("上传失败：后端没响应或超时（首次上传要加载模型，请耐心等待）")
        return None
    if res.status_code != 200:
        st.error(f"上传失败：{backend_error_message(res)}")
        return None
    return res.json()


def delete_knowledge_doc(doc_id: str) -> bool:
    """删除文档（后端会同步清掉表与向量库）。"""
    try:
        res = requests.delete(f"{BACKEND_KNOWLEDGE_URL}/{doc_id}", timeout=30)
    except (requests.ConnectionError, requests.Timeout):
        st.error("删除失败：后端没响应")
        return False
    if res.status_code != 200:
        st.error(f"删除失败：{backend_error_message(res)}")
        return False
    return True


def render_chunk_preview(doc_id: str):
    """按需拉取某个文档的分块，展示页码/章节定位。"""
    try:
        res = requests.get(f"{BACKEND_KNOWLEDGE_URL}/{doc_id}", timeout=5)
    except (requests.ConnectionError, requests.Timeout):
        st.caption("（连不上后端）")
        return
    if res.status_code != 200:
        st.caption(f"读取分块失败（{res.status_code}）")
        return

    doc = res.json()
    st.divider()
    st.caption(f"分块预览 · {doc['title']}")
    if not doc["chunks"]:
        st.caption("这个文档没有分块（可能是扫描件）")
        return
    for c in doc["chunks"]:
        loc = []
        if c.get("page_no"):
            loc.append(f"第 {c['page_no']} 页")
        if c.get("section_path"):
            loc.append(c["section_path"])
        st.markdown(f"`#{c['chunk_idx']}` " + (" · ".join(loc) if loc else "无定位"))
        text = c["content"].replace("\n", " ")
        st.caption(text[:160] + ("…" if len(text) > 160 else ""))


with st.sidebar:
    if st.button("+ 新建会话", use_container_width=True, type="primary"):
        new_session()
        st.rerun()

    # ========== 知识库面板 ==========
    with st.expander("📚 知识库", expanded=False):
        st.caption("上传文档后，AI 会根据资料回答")
        uploaded = st.file_uploader(
            "上传文档",
            type=["pdf", "docx", "xlsx", "xlsm", "md", "markdown", "txt", "html", "htm"],
            help="支持 PDF / Word(.docx) / Excel(.xlsx) / Markdown / txt / HTML；扫描版 PDF 自动 OCR；老版 .doc/.xls 请先另存为新格式",
        )
        if uploaded is not None and st.button(
            "上传到知识库", use_container_width=True, type="primary", key="upload-file"
        ):
            data = upload_knowledge_file(uploaded)
            if data is not None:
                if data.get("duplicated"):
                    st.info("内容相同，这篇文档已经在知识库里了")
                elif data.get("needs_ocr"):
                    st.warning("疑似扫描件，提取不到文字，暂无法入库检索")
                else:
                    st.success(f"上传成功！切成 {data['chunk_count']} 块")
                st.rerun()

        st.divider()
        st.caption("已上传的文档")
        try:
            res_docs = requests.get(BACKEND_KNOWLEDGE_URL, timeout=5)
        except (requests.ConnectionError, requests.Timeout):
            st.caption("（连不上后端）")
        else:
            if res_docs.status_code != 200:
                st.caption(f"读取失败（{res_docs.status_code}）")
            else:
                docs = res_docs.json()["docs"]
                if not docs:
                    st.caption("知识库还是空的")
                for d in docs:
                    meta = d["source_type"]
                    if d.get("page_count"):
                        meta += f" · {d['page_count']} 页"
                    meta += f" · {d['chunk_count']} 块"
                    if d.get("status") != "ready":
                        meta += f" · {d['status']}"

                    cols = st.columns([5, 1, 1])
                    with cols[0]:
                        st.markdown(f"**{d['title']}**")
                        st.caption(meta)
                    with cols[1]:
                        if st.button("🔍", key=f"peek-{d['doc_id']}", help="查看分块"):
                            opened = st.session_state.get("peek_doc") == d["doc_id"]
                            st.session_state["peek_doc"] = None if opened else d["doc_id"]
                            st.rerun()
                    with cols[2]:
                        if st.button("🗑", key=f"del-doc-{d['doc_id']}", help="删除此文档"):
                            if delete_knowledge_doc(d["doc_id"]):
                                if st.session_state.get("peek_doc") == d["doc_id"]:
                                    st.session_state["peek_doc"] = None
                                st.rerun()

                if st.session_state.get("peek_doc"):
                    render_chunk_preview(st.session_state["peek_doc"])

    st.divider()

    sessions = fetch_sessions()
    if not sessions:
        st.caption("还没有会话，发第一句话开始聊天")
    else:
        for s in sessions:
            sid = s["session_id"]
            is_active = sid == st.session_state.session_id
            cols = st.columns([4, 1])
            with cols[0]:
                # type= 不同可以一眼看出哪个是当前会话
                btn_type = "primary" if is_active else "secondary"
                if st.button(s["title"], key=f"go-{sid}", use_container_width=True, type=btn_type):
                    switch_session(sid)
                    st.rerun()
            with cols[1]:
                if st.button("🗑", key=f"del-{sid}", help="删除此会话"):
                    delete_session(sid)
                    st.rerun()


# ============== 主区域：聊天 ==============

# 加载当前会话的历史（首次或切换后都走这里）
need_load = "messages" not in st.session_state or st.session_state.messages is None
if need_load:
    st.session_state.messages = []
    try:
        res = requests.get(f"{BACKEND_HISTORY_URL}/{st.session_state.session_id}", timeout=5)
    except (requests.ConnectionError, requests.Timeout):
        st.error("连不上后端：请先启动 uvicorn backend:app --reload --port 8000")
    else:
        if res.status_code == 200:
            st.session_state.messages = res.json()["messages"]
        else:
            st.error(f"加载历史失败（{res.status_code}）：{backend_error_message(res)}")

# 渲染历史消息
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 空会话：欢迎页 + 示例问题卡片（点按钮等价于在输入框敲了这句话）
if not st.session_state.messages:
    st.markdown(
        """
        <div style="padding: 3rem 1.2rem 0.5rem 1.2rem;">
          <h2 style="font-weight:700; margin-bottom:0.4rem;">👋 你好，我是本地知识库 AI 问答助手</h2>
          <p style="color:#64748b; margin-bottom:0;">
            本地知识库问答助手 · 答案附章节级引用 · 检索过程全程可见
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("试试点下面的问题开始：")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"example-{q}", use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()

# 输入框
user_input = st.chat_input("请输入你的问题")
# 点了示例卡片时没有手敲输入，从 session_state 取回这句话，走同一条提问流程
if not user_input:
    user_input = st.session_state.pop("pending_question", None)

if user_input:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="🤖"):
        try:
            # st.write_stream 吃一个生成器，自动实现打字机效果，返回完整文本
            ai_reply = st.write_stream(stream_ai_reply(st.session_state.session_id, user_input))
        except (requests.ConnectionError, requests.Timeout):
            st.error("后端没响应：请确认 uvicorn 还在运行，然后重发这句话")
            st.stop()
        except RuntimeError as e:
            # 流开始前就失败（有状态码的错误）
            st.error(str(e))
            st.stop()

        # 思考过程：推理模型先思考再回答，把思考链折叠展示，让用户知道刚才那十几秒在干嘛
        reasoning = getattr(stream_ai_reply, "reasoning", "")
        if reasoning:
            with st.expander("💭 AI 思考过程", expanded=False):
                st.caption(reasoning.strip())

        # 引用来源：检索到了资料就展示，让用户知道答案依据哪篇文档哪一章
        citations = getattr(stream_ai_reply, "citations", [])
        if citations:
            st.markdown("---")
            st.caption("📎 引用来源")
            for c in citations:
                loc = [f"**{c['title']}**"]
                if c.get("section_path"):
                    loc.append(c["section_path"])
                if c.get("page_no"):
                    loc.append(f"第 {c['page_no']} 页")
                st.markdown(f"[{c['ref']}] {' · '.join(loc)}")

        # 检索过程面板：把「提问 → 双路召回 → 融合 → 精排 → 送入 LLM」这条链路
        # 逐步摊开，让用户看到答案到底是从哪些块里捞出来的
        trace = getattr(stream_ai_reply, "trace", None)
        if trace:
            with st.expander("🔍 本次检索过程", expanded=False):
                n_dense = len(trace.get("dense_raw", []))
                n_bm25 = len(trace.get("bm25", []))
                n_final = len(trace.get("final", []))
                chain = f"知识库共 {trace.get('total_chunks', '?')} 块 ｜ 向量召回 {n_dense} 条"
                if trace.get("rerank_on"):
                    chain += " → Cross-encoder 精排"
                chain += f" ｜ BM25 字面召回 {n_bm25} 条 → RRF 融合 → 最终送入 LLM {n_final} 条"
                st.caption(chain)

                steps = [
                    ("① 向量召回", trace.get("dense_raw", [])),
                    ("② 精排后", trace.get("dense_reranked", [])),
                    ("③ BM25 召回", trace.get("bm25", [])),
                    ("④ RRF 融合", trace.get("fused", [])),
                    ("⑤ 送入 LLM", trace.get("final", [])),
                ]
                tabs = st.tabs([f"{name}({len(items)})" for name, items in steps])
                for tab, (_, items) in zip(tabs, steps):
                    with tab:
                        if not items:
                            st.caption("（此路已关闭或无召回）")
                            continue
                        for i, it in enumerate(items, 1):
                            loc = f"**{it.get('title', '')}**"
                            if it.get("section_path"):
                                loc += f" · {it['section_path']}"
                            st.markdown(f"`#{i}` " + loc)
                            st.caption((it.get("snippet") or "")[:110])

    # 流结束后才归档；如果回复里含 [错误] 标记（流中间出的错），不入库
    if "[错误]" not in ai_reply:
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
