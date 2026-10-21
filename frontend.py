import json
import uuid

import requests
import streamlit as st

# 与后端 chat.py 的 CITATION_MARKER 保持一致
CITATION_MARKER = "__CITATIONS_JSON__"

BACKEND_CHAT_URL = "http://127.0.0.1:8000/api/chat"
BACKEND_HISTORY_URL = "http://127.0.0.1:8000/api/history"
BACKEND_SESSIONS_URL = "http://127.0.0.1:8000/api/sessions"
BACKEND_KNOWLEDGE_URL = "http://127.0.0.1:8000/api/knowledge"


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

    同时检测引用标记 __CITATIONS_JSON__：标记前的文本正常 yield 给打字机，
    标记后的 JSON 解析后存到 stream_ai_reply.citations 供调用方展示。
    这样 st.write_stream 只显示干净的回复文本，引用 JSON 不会闪到界面上。
    """
    res = requests.post(
        BACKEND_CHAT_URL,
        json={"session_id": session_id, "message": message},
        stream=True,
        timeout=120,
    )
    if res.status_code != 200:
        raise RuntimeError(f"出错了（{res.status_code}）：{backend_error_message(res)}")

    buffer = ""
    citations = []
    in_citations = False
    citation_buffer = ""
    marker_len = len(CITATION_MARKER)

    for chunk in res.iter_content(decode_unicode=True):
        if not chunk:
            continue

        if in_citations:
            # 标记已出现，后续都是引用 JSON，不再 yield
            citation_buffer += chunk
            continue

        buffer += chunk

        if CITATION_MARKER in buffer:
            # 找到标记：yield 标记前的文本，切到引用模式
            text, _, rest = buffer.partition(CITATION_MARKER)
            if text:
                yield text
            in_citations = True
            citation_buffer = rest
            buffer = ""
        elif len(buffer) > marker_len:
            # 持有末尾 marker_len 个字符防止标记被截断，其余安全 yield
            safe = buffer[:-marker_len]
            yield safe
            buffer = buffer[-marker_len:]

    # 没有标记的余量直接 yield（知识库为空或检索无结果时不会有标记）
    if not in_citations and buffer:
        yield buffer

    # 解析引用 JSON 供调用方使用
    if citation_buffer:
        try:
            citations = json.loads(citation_buffer.strip())
        except json.JSONDecodeError:
            pass

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
            type=["pdf", "docx", "md", "markdown", "txt", "html", "htm"],
            help="支持 PDF / Word(.docx) / Markdown / txt / HTML；老版 .doc 请先另存为 .docx",
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
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
user_input = st.chat_input("请输入你的问题")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
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

    # 流结束后才归档；如果回复里含 [错误] 标记（流中间出的错），不入库
    if "[错误]" not in ai_reply:
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
