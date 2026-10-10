"""对话历史读写。"""

from app.db.connection import get_conn


def save_message(session_id: str, role: str, content: str):
    conn = get_conn()
    # ? 占位符：值永远只当数据用，不会被当成 SQL 代码执行（防注入，别用 f-string 拼 SQL）
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    # commit = 真正写进文件；不 commit，进程退出数据就没了
    conn.commit()
    conn.close()


def load_history(session_id: str) -> list:
    conn = get_conn()
    # ORDER BY id：按插入顺序还原对话顺序
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def list_sessions() -> list:
    """列出所有会话：取每个会话的第一条用户消息当标题，按最近更新倒序。

    ROW_NUMBER 先在会话内按 id 排序，再取第 1 条（即第一句用户问题）。
    """
    conn = get_conn()
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
        {
            "session_id": r["session_id"],
            "title": r["content"][:20] + ("…" if len(r["content"]) > 20 else ""),
            "updated_at": r["last_time"],
        }
        for r in rows
    ]


def delete_session(session_id: str) -> int:
    """删除一个会话的所有消息，返回删了多少条。"""
    conn = get_conn()
    cur = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    return deleted
