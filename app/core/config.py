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

# 人设：基于用户上传的资料问答。
# 「有资料就依据资料、没答案就说不知道」这类约束在 rag.py 组装提示时追加，
# 因为它只在检索到资料时才该生效，放在这里会变成无条件约束。
SYSTEM_PROMPT = (
    "你是企业知识库助手，依据用户上传的文档资料回答问题。"
    "回答要准确、简洁，使用中文，不要输出与问题无关的内容。"
)
