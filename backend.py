"""后端入口。

启动方式保持不变：
    uvicorn backend:app --reload

真正的实现在 app/ 包里，这里只做一层壳，
这样既能让代码按模块拆开，又不用改现有的启动命令和 PyCharm 运行配置。
"""

from app.main import app

__all__ = ["app"]
