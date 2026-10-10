"""本地 embedding。用 sentence-transformers，完全免费，不依赖外部 API。"""

import time

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.core.config import RAG_EMBEDDING_MODEL
from app.core.logging import logger


class LocalEmbeddings(Embeddings):
    """实现 LangChain 的 Embeddings 接口，方便直接喂给 FAISS。"""

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


_instance: LocalEmbeddings | None = None


def get_embeddings() -> LocalEmbeddings:
    """懒加载：第一次用到时才下载/加载模型，避免拖慢进程启动。"""
    global _instance
    if _instance is None:
        logger.info("正在加载本地 embedding 模型：%s", RAG_EMBEDDING_MODEL)
        started = time.perf_counter()
        _instance = LocalEmbeddings(RAG_EMBEDDING_MODEL)
        logger.info(
            "embedding 模型加载完成 耗时=%.0fms",
            (time.perf_counter() - started) * 1000,
        )
    return _instance
