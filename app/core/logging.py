"""统一日志。全项目只在这里创建 logger，其他模块从这里导入。"""

import logging
import sys

from app.core.config import LOG_LEVEL

# 自带 handler 且 propagate=False：不去动 uvicorn 自己的日志配置，
# 避免访问日志被打印两遍。
logger = logging.getLogger("aichat")
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
logger.propagate = False


def preview(text: str, limit: int = 60) -> str:
    """日志里只留开头一小段，避免把整篇文档或长问题写进日志。"""
    one_line = " ".join(text.split())
    return one_line[:limit] + ("…" if len(one_line) > limit else "")
