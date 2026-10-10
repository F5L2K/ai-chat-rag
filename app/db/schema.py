"""建表。所有表结构集中在这里，加字段只改这个文件。

IF NOT EXISTS：文件已存在就什么都不做，所以反复重启是安全的。
注意它不会修改已存在的表 —— 已建过的表要加字段，得手动 ALTER 或删库重建。
"""

from app.core.config import DB_PATH
from app.core.logging import logger
from app.db.connection import get_conn

TABLES = [
    # 对话消息
    """
    CREATE TABLE IF NOT EXISTS messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # 文档主表：一行一篇文档
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id       TEXT PRIMARY KEY,
        title        TEXT NOT NULL,
        source_type  TEXT NOT NULL DEFAULT 'text',
        source_path  TEXT,
        content_hash TEXT NOT NULL,
        version      INTEGER NOT NULL DEFAULT 1,
        page_count   INTEGER,
        chunk_count  INTEGER NOT NULL DEFAULT 0,
        status       TEXT NOT NULL DEFAULT 'ready',
        error_msg    TEXT,
        tenant_id    TEXT NOT NULL DEFAULT 'default',
        dept_id      TEXT,
        acl_tags     TEXT,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # 分块明细：page_no / section_path 是「带引用回答」的地基，
    # 解析时没存下来，事后补不回来（切块后文字与页码的对应关系就丢了）
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id     TEXT PRIMARY KEY,
        doc_id       TEXT NOT NULL,
        chunk_idx    INTEGER NOT NULL,
        content      TEXT NOT NULL,
        page_no      INTEGER,
        section_path TEXT,
        char_start   INTEGER,
        char_end     INTEGER,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
    )
    """,
    # 按文档查分块是高频操作
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)",
    # 上传时按 hash 查重
    "CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(content_hash)",
]


def init_db():
    conn = get_conn()
    for stmt in TABLES:
        conn.execute(stmt)
    conn.commit()
    conn.close()
    logger.info("SQLite 初始化完成 path=%s", DB_PATH)
