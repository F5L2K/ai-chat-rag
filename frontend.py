import uuid

import requests
import streamlit as st

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
    yield 一次 = 吐出几个字；st.write_stream 会把它们逐字打到界面上。"""
    res = requests.post(
        BACKEND_CHAT_URL,
        json={"session_id": session_id, "message": message},
        stream=True,   # 关键：不等到全部下载完，来一块拿一块
        timeout=120,   # 流式生成可能比较久，给足两分钟
    )
    # 流开始前就出错（如 422、500）：走老的错误契约
    if res.status_code != 200:
        raise RuntimeError(f"出错了（{res.status_code}）：{backend_error_message(res)}")
    # 正常流式：逐块推送
    for chunk in res.iter_content(decode_unicode=True):
        if chunk:
            yield chunk


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


with st.sidebar:
    if st.button("+ 新建会话", use_container_width=True, type="primary"):
        new_session()
        st.rerun()

    # ========== 知识库面板 ==========
    with st.expander("📚 知识库", expanded=False):
        st.caption("上传文档后，AI 会根据资料回答")
        doc_title = st.text_input("文档标题", placeholder="比如：员工手册")
        doc_content = st.text_area(
            "文档内容",
            placeholder="把要让 AI 学习的资料粘贴到这里…",
            height=120,
        )
        if st.button("上传到知识库", use_container_width=True, disabled=not (doc_title and doc_content)):
            try:
                res = requests.post(
                    f"{BACKEND_KNOWLEDGE_URL}/upload",
                    json={"title": doc_title, "content": doc_content},
                    timeout=60,  # embedding 可能比较慢
                )
                if res.status_code == 200:
                    st.success(f"上传成功！切成了 {res.json()['chunk_count']} 块")
                    st.rerun()
                else:
                    st.error(f"上传失败：{backend_error_message(res)}")
            except (requests.ConnectionError, requests.Timeout):
                st.error("上传失败：后端没响应或超时")

        st.divider()
        st.caption("已上传的文档")
        try:
            res_docs = requests.get(BACKEND_KNOWLEDGE_URL, timeout=5)
            if res_docs.status_code == 200:
                docs = res_docs.json()["docs"]
                if not docs:
                    st.caption("知识库还是空的")
                else:
                    for d in docs:
                        st.write(f"**{d['title']}** · {d['chunk_count']} 块")
        except (requests.ConnectionError, requests.Timeout):
            st.caption("（连不上后端）")

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

    # 流结束后才归档；如果回复里含 [错误] 标记（流中间出的错），不入库
    if "[错误]" not in ai_reply:
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
