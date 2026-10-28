"""FastAPI 应用装配：中间件、异常兜底、路由注册。

这是后端的唯一入口，backend.py 只是它的一层壳。
"""

import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import logger
from app.db.schema import init_db
from app.routers import chat, knowledge

app = FastAPI(title="AI聊天后端接口", version="1.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """给每个请求分配一个短 ID，记录访问结果和耗时。

    注意：流式接口（/api/chat）在这里量到的是「首字节耗时」，
    完整生成耗时由 chat 接口内部单独记录。
    """
    request_id = uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "[%s] %s %s 抛出未捕获异常 耗时=%.0fms",
            request_id, request.method, request.url.path,
            (time.perf_counter() - started) * 1000,
        )
        raise
    logger.info(
        "[%s] %s %s -> %d 耗时=%.0fms",
        request_id, request.method, request.url.path, response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def safety_net(request: Request, exc: Exception):
    """全局兜底网：没被上面接住的异常都在这里收场。

    完整堆栈打到后端终端（给开发者排错），前端只收到一句干净的 JSON（给用户看）。
    """
    request_id = getattr(request.state, "request_id", "-")
    logger.error(
        "[%s] 未捕获异常 %s %s",
        request_id, request.method, request.url.path,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "internal_error", "message": "服务器内部错误，请稍后重试"}},
    )


app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["knowledge"])

# 模块加载时执行一次：进程启动 → 确保表存在
init_db()


def _warmup_models():
    """后台预热：在后台线程里把 embedding / 向量库 / reranker / BM25 都加载好。

    不阻塞 uvicorn 启动：后端端口立刻能接请求，但如果第一个问答请求
    在预热完成前进来，仍会等模型加载完（与原来懒加载行为一致）。
    预热完成后，所有请求都走热缓存，不再有 35 秒冷启动等待。
    """
    try:
        from app.services.vector_store import get_vector_store
        from app.services.rerank import _get_model as _get_reranker
        from app.services.bm25 import search as _bm25_search

        t0 = time.perf_counter()
        logger.info("后台预热开始：加载 embedding + 向量库 + reranker + BM25...")
        get_vector_store()          # 内部会触发 embedding 加载 + FAISS 读盘
        _get_reranker()            # 触发 cross-encoder 加载
        _bm25_search("预热探针", 1)  # 触发 BM25 索引构建
        logger.info("后台预热完成，耗时=%.0fms", (time.perf_counter() - t0) * 1000)
    except Exception as e:
        # 预热失败不影响启动：第一个请求会走懒加载兜底
        logger.warning("后台预热失败（%s），将退回懒加载", e)


# daemon 线程：主进程退出时跟着退，不阻塞关机
threading.Thread(target=_warmup_models, daemon=True, name="model-warmup").start()
