import os
import sqlite3
import traceback

# HuggingFace 国内镜像（必须在 import sentence_transformers 之前设置，
# 否则模型下载会直连 huggingface.co 导致超时）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
    APIStatusError,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ========== RAG 相关导入 ==========
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# 加载环境变量
load_dotenv()

# 初始化大模型客户端
client = OpenAI(
    api_key=os.environ.get("dp_api_key"),
    base_url="https://api.deepseek.com"
)

# 创建FastAPI实例
app = FastAPI(title="AI聊天后端接口", version="1.0")

# ==========四阶段：SQLite 持久化（状态搬进硬盘）==========
SYSTEM_PROMPT = "你是严谨的数学助手，写清楚计算步骤"
# 数据库就是项目目录下的一个普通文件，跟着代码走，重启后端也不丢
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.db")


def init_db():
    # 建表。IF NOT EXISTS：文件已存在就什么都不做，所以反复重启是安全的
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# 模块加载时执行一次：进程启动 → 确保表存在
init_db()


# ==========RAG：知识库与向量检索==========
RAG_CHUNK_SIZE = 500       # 每块多少字
RAG_CHUNK_OVERLAP = 50     # 块之间重叠多少字（防止切块把意思切断）
RAG_TOP_K = 3              # 每次检索取最相关的几块
FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_store")
RAG_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 本地中文 embedding 模型（~100MB）


class LocalEmbeddings(Embeddings):
    """用本地 sentence-transformers 模型做向量化，完全免费，不依赖 API。
    实现 LangChain 的 Embeddings 接口，方便直接喂给 FAISS。"""

    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（上传文档时用）。"""
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """单句向量化（检索时用）。"""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


# 懒加载：embeddings 初始为 None，第一次用到时才下载/加载模型
_embeddings_instance: LocalEmbeddings | None = None


def get_embeddings() -> LocalEmbeddings:
    """获取 embedding 模型实例，第一次调用时才加载（可能需要下载模型）。"""
    global _embeddings_instance
    if _embeddings_instance is None:
        print(f"[RAG] 正在加载本地 embedding 模型：{RAG_EMBEDDING_MODEL}")
        _embeddings_instance = LocalEmbeddings(RAG_EMBEDDING_MODEL)
        print("[RAG] 模型加载完成")
    return _embeddings_instance

# 文本切块器：按段落→句子→字符逐级切，尽量保持语义完整
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=RAG_CHUNK_SIZE,
    chunk_overlap=RAG_CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)

# 知识文档元数据表（记录上传了哪些文档，方便列表展示）
def init_knowledge_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_docs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_knowledge_db()


def init_vector_store() -> FAISS:
    """加载或创建 FAISS 向量库。
    如果本地已有索引文件就加载；没有就新建一个空的。
    如果旧索引加载失败（比如换了 embedding 模型导致维度不匹配），自动删掉重建。"""
    emb = get_embeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            return FAISS.load_local(
                FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"[RAG] 旧向量库加载失败（{e}），将删除重建")
            import shutil
            shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)
    # 新建空库：先拿一句样例文本测一下维度，再用这个维度建索引
    sample = emb.embed_query("init")
    dimension = len(sample)
    index = faiss.IndexFlatL2(dimension)
    return FAISS(
        embedding_function=emb,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


# 全局向量库实例（懒初始化：None = 还没建，第一次上传文档时才初始化）
vector_store: FAISS | None = None


def get_vector_store() -> FAISS:
    """获取向量库实例，需要时才初始化。"""
    global vector_store
    if vector_store is None:
        vector_store = init_vector_store()
    return vector_store


def save_vector_store():
    """把当前向量库保存到磁盘。每次新增文档后调用。"""
    get_vector_store().save_local(FAISS_INDEX_PATH)


def add_document(title: str, content: str) -> int:
    """把一篇文档切块、向量化、加入向量库，返回切块数量。"""
    # 1. 切块
    chunks = text_splitter.split_text(content)
    if not chunks:
        return 0
    # 2. 转成 Document 对象（带上元数据，方便以后知道哪块来自哪篇文档）
    docs = [
        Document(page_content=chunk, metadata={"title": title, "chunk_idx": i})
        for i, chunk in enumerate(chunks)
    ]
    # 3. 加入向量库（内部会自动调 embedding 模型向量化）
    get_vector_store().add_documents(docs)
    # 4. 落盘（持久化）
    save_vector_store()
    # 5. 元数据入库
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO knowledge_docs (title, chunk_count) VALUES (?, ?)",
        (title, len(chunks)),
    )
    conn.commit()
    conn.close()
    return len(chunks)


def list_knowledge() -> list:
    """列出所有已上传的文档。"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, title, chunk_count, created_at FROM knowledge_docs ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [
        {"id": rid, "title": title, "chunk_count": cc, "created_at": t}
        for rid, title, cc, t in rows
    ]


def retrieve_relevant(query: str) -> list:
    """根据问题检索最相关的文本片段，返回字符串列表。
    如果向量库还没初始化或为空，返回空列表。"""
    if vector_store is None:
        return []
    if get_vector_store().index.ntotal == 0:
        return []
    docs = get_vector_store().similarity_search(query, k=RAG_TOP_K)
    return [doc.page_content for doc in docs]


def build_rag_prompt(query: str) -> str:
    """把检索到的资料拼进 system prompt，告诉模型"根据以下资料回答"。"""
    chunks = retrieve_relevant(query)
    if not chunks:
        return SYSTEM_PROMPT
    # 把相关片段拼起来，用明确的分隔标记
    context = "\n\n---\n\n".join(f"资料片段 {i+1}：\n{chunk}" for i, chunk in enumerate(chunks))
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"以下是可能相关的参考资料，请根据资料回答用户问题。"
        f"如果资料里没有答案，就如实说不知道，不要编造。\n\n"
        f"===== 参考资料 =====\n"
        f"{context}\n"
        f"===== 参考资料结束 ====="
    )


def save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    # ? 占位符：值永远只当数据用，不会被当成 SQL 代码执行（防注入，别用 f-string 拼 SQL）
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    # commit = 真正写进文件；不 commit，进程退出数据就没了
    conn.commit()
    conn.close()


def load_history(session_id: str) -> list:
    conn = sqlite3.connect(DB_PATH)
    # ORDER BY id：按插入顺序还原对话顺序
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]


def list_sessions() -> list:
    """列出所有会话：取每个 session_id 的第一条用户消息当标题，按最近更新倒序。
    用窗口函数 ROW_NUMBER：先在每个会话内按 id 排序，再取第 1 条（即第一句用户问题）。"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT session_id, content, last_time
        FROM (
            SELECT
                session_id,
                content,
                ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY id) AS rn,
                MAX(created_at) OVER (PARTITION BY session_id)   AS last_time
            FROM messages
            WHERE role = 'user'
        )
        WHERE rn = 1
        ORDER BY last_time DESC
    """).fetchall()
    conn.close()
    # 标题截断到 20 字，太长侧边栏显示不下
    return [
        {"session_id": sid, "title": content[:20] + ("…" if len(content) > 20 else ""), "updated_at": t}
        for sid, content, t in rows
    ]


def delete_session(session_id: str) -> int:
    """删除一个会话的所有消息，返回删了多少条。"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    return deleted


# ==========Pydantic 数据模型（契约）==========
class ChatRequest(BaseModel):
    session_id: str
    message: str


class DocUploadRequest(BaseModel):
    title: str
    content: str


# ==========POST /api/chat：流式输出（逐 token 推送，边生成边返回）==========
# 注意：流式接口一旦开始推送就没法回头改状态码了，
# 所以"流开始前的错误"用 HTTPException（有状态码），
# "流开始后的错误"直接在流里塞一段错误文字（状态码已经是 200）。
@app.post("/api/chat")
async def chat(req: ChatRequest):
    # ===== 流开始前：校验 + 读历史 + 存用户消息 =====
    # 这些步骤出错可以正常返回 4xx/5xx + JSON 错误
    history = load_history(req.session_id)
    save_message(req.session_id, "user", req.message)
    history.append({"role": "user", "content": req.message})

    # RAG：根据当前问题检索相关资料，动态生成 system prompt
    # （只有最新这句问题用来检索，因为检索是找"跟此刻问题相关的资料"）
    system_with_rag = build_rag_prompt(req.message)
    full_messages = [{"role": "system", "content": system_with_rag}] + history

    # ===== 生成器：真正的流式推送在这里 =====
    # yield 一次 = 推一块给前端；前端收到一块就显示一块
    def generate():
        full_reply = ""
        try:
            stream = client.chat.completions.create (
                model="deepseek-v4-pro",
                messages=full_messages,
                stream=True,  # 关键：流式模式，返回的是迭代器不是完整响应
            )
            for chunk in stream:
                # 每个 chunk 里可能有 delta.content（增量的几个字），也可能没有
                delta = chunk.choices[0].delta
                if delta.content:
                    full_reply += delta.content
                    yield delta.content  # 把这几个字立刻推给前端
        except APITimeoutError:
            yield "\n\n[错误] 模型响应超时，请稍后重试"
            return
        except APIConnectionError:
            yield "\n\n[错误] 连不上 DeepSeek 服务，请稍后重试"
            return
        except RateLimitError:
            yield "\n\n[错误] 请求太频繁，请稍等几秒再试"
            return
        except AuthenticationError:
            traceback.print_exc()
            yield "\n\n[错误] 服务配置有误，请联系管理员"
            return
        except APIStatusError:
            yield "\n\n[错误] DeepSeek 服务返回错误，请稍后重试"
            return
        except Exception:
            # 兜底：流中间的任何意外
            traceback.print_exc()
            yield "\n\n[错误] 生成过程中出现异常，请稍后重试"
            return

        # 流正常结束：把完整回复落库
        save_message(req.session_id, "assistant", full_reply)

    # text/event-stream 是 SSE 标准 MIME，浏览器/客户端知道这是流式响应
    return StreamingResponse(generate(), media_type="text/event-stream")


# ==========GET /api/history：从数据库查历史==========
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    # system 人设从未入库（它是配置不是数据），查出来的天然就是可见消息
    return {"session_id": session_id, "messages": load_history(session_id)}


# ==========GET /api/sessions：列出所有会话==========
@app.get("/api/sessions")
async def list_all_sessions():
    return {"sessions": list_sessions()}


# ==========DELETE /api/sessions/{session_id}：删除一个会话==========
@app.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    deleted = delete_session(session_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail={
            "code": "session_not_found",
            "message": "会话不存在",
        })
    return {"deleted": deleted}


# ==========知识库接口==========
@app.post("/api/knowledge/upload")
async def upload_doc(req: DocUploadRequest):
    """上传一篇文档到知识库：切块 → 向量化 → 存入 FAISS。
    返回切块数量。embedding 调用在这里，费用由上传触发。"""
    if not req.title.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail={
            "code": "invalid_doc",
            "message": "标题和内容都不能为空",
        })
    try:
        chunk_count = add_document(req.title, req.content)
    except Exception as e:
        # embedding 失败（如 key 无效、网络问题）
        raise HTTPException(status_code=500, detail={
            "code": "embedding_failed",
            "message": f"文档向量化失败：{str(e)}",
        })
    return {"title": req.title, "chunk_count": chunk_count}


@app.get("/api/knowledge")
async def list_docs():
    """列出知识库中所有文档。"""
    return {"docs": list_knowledge()}


# ==========全局兜底网：没被上面接住的任何异常，都到这里收场==========
# 完整堆栈打到后端终端（给开发者排错），前端只收到一句干净的 JSON（给用户看）
@app.exception_handler(Exception)
async def safety_net(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "internal_error", "message": "服务器内部错误，请稍后重试"}},
    )
