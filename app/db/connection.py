"""SQLite 连接。所有数据库访问都走 get_conn()，
以后要换连接池或换库，只需要改这一个文件。
"""

import sqlite3

from app.core.config import DB_PATH


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    # Row 让结果支持按列名访问（row["title"]），比元组下标可读
    conn.row_factory = sqlite3.Row
    # SQLite 默认不校验外键，ON DELETE CASCADE 要靠这个开关才生效
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
