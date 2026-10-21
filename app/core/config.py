"""集中配置：路径、密钥、RAG 参数。

所有模块只从这里读配置，不要在别处直接读 os.environ，
否则以后换部署方式（容器、多环境）会到处漏风。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# app/core/config.py → app/core → app → 项目根
BASE_DIR = Path(__file__).resolve().parents[2]

# ==========模型密钥==========
# DEEPSEEK_API_KEY 是规范名；dp_api_key 是历史变量名，.env 改名后可删掉后半截
API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("dp_api_key")
if not API_KEY:
    raise RuntimeError(
        "未配置模型密钥：请在 .env 中设置 DEEPSEEK_API_KEY（可参考 .env.example）"
    )

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHAT_MODEL = "deepseek-v4-pro"

# ==========日志==========
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# ==========存储==========
DB_PATH = BASE_DIR / "chat.db"
FAISS_INDEX_PATH = BASE_DIR / "faiss_store"

# ==========RAG==========
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 3
RAG_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# ==========重排序（两阶段检索的第二阶段）==========
# 第一阶段用 FAISS 向量召回 RAG_RECALL_K 个候选（要求 >= RAG_TOP_K），
# 第二阶段用 Cross-encoder 精排出 RAG_TOP_K 个喂给模型。
# 召回归召回、精排归精排：召回池给足，避免正确块在第一阶段就被挤掉。
RAG_RECALL_K = 20
# 用环境变量可关闭重排序（RAG_RERANK_ENABLED=0），方便对比与排障
RAG_RERANK_ENABLED = os.environ.get("RAG_RERANK_ENABLED", "1") != "0"
# 模型可用环境变量覆盖为本地目录路径（离线环境 / 已自行下载时用）
RAG_RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")

# ==========混合检索（BM25 + 向量）==========
# 向量把编号、错误码、型号这类「无语义、只靠字面区分」的串平滑掉
# （E102 与 E202 在向量空间几乎重合），BM25 是纯字面词频匹配，正好补这一刀：
# 两路各召回一批候选，用 RRF 融合后交给精排。设 RAG_HYBRID_ENABLED=0 可关闭。
RAG_HYBRID_ENABLED = os.environ.get("RAG_HYBRID_ENABLED", "1") != "0"
# BM25 单路召回数，与向量召回同量级，融合才均衡
RAG_BM25_TOP_K = 20
# RRF 融合常数：score = Σ 1/(RAG_RRF_K + rank)。60 取自原论文默认值，
# 作用是压平头部，让「两路都靠前」胜过「单路第一」，避免单路噪声通吃。
RAG_RRF_K = 60

# 人设：基于用户上传的资料问答。
# 「有资料就依据资料、没答案就说不知道」这类约束在 rag.py 组装提示时追加，
# 因为它只在检索到资料时才该生效，放在这里会变成无条件约束。
SYSTEM_PROMPT = (
    "你是企业知识库助手，依据用户上传的文档资料回答问题。"
    "回答要准确、简洁，使用中文，不要输出与问题无关的内容。"
)
