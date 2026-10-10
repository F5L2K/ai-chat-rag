# AIChat - 带知识库的 AI 聊天助手

基于 **FastAPI + Streamlit + DeepSeek + RAG** 的本地 AI 聊天助手。支持多会话聊天、流式输出，以及上传 PDF / Word / Markdown / 文本构建本地知识库（向量检索问答）。

## 功能特性

- 💬 **多会话聊天**：会话自动保存到本地 SQLite，刷新页面不丢失，支持新建 / 切换 / 删除
- ⚡ **流式输出**：AI 回复逐字显示，无需等待完整生成
- 📚 **多格式知识库**：支持 PDF / Word(.docx) / Markdown / txt 上传，自动解析、切块、向量化并建索引
- 🔖 **页码与章节定位**：解析时保留每块的来源页码（PDF）与章节路径（三种格式），为后续「带引用回答」打底
- 🧠 **本地向量化**：使用 `BAAI/bge-small-zh-v1.5` 本地 embedding 模型，不依赖额外 API；索引懒加载并落盘，重启后自动恢复
- 🧹 **文档管理**：列表查看、分块预览、删除文档（数据库与向量库同步清理，不会残留漂移数据）
- 🔁 **重复检测**：按内容指纹（SHA-256）识别重复文档，相同内容不重复入库
- 🇨🇳 **国内镜像**：模型下载自动走 HuggingFace 国内镜像（`hf-mirror.com`），无需科学上网

## 技术栈

| 部分 | 技术 |
|---|---|
| 后端 | Python · FastAPI · Uvicorn |
| 前端 | Streamlit |
| 大模型 | DeepSeek（OpenAI 兼容接口） |
| 文档解析 | pdfplumber（PDF）· python-docx（Word）· 正则（Markdown / txt） |
| 文本切分 | 章节 / 页码感知切分 + LangChain RecursiveCharacterTextSplitter（500/50） |
| 向量检索 | LangChain · FAISS · sentence-transformers |
| 存储 | SQLite（messages / documents / chunks 三表）· 本地向量索引（faiss_store/） |

## 目录结构

```
AIChat/
├── backend.py            # 后端入口壳：暴露 app，启动命令仍是 uvicorn backend:app
├── frontend.py           # Streamlit 前端界面
├── app/
│   ├── main.py           # FastAPI 装配：中间件、异常兜底、路由注册
│   ├── core/             # config.py（配置与密钥校验）· logging.py
│   ├── db/               # connection.py · schema.py（messages / documents / chunks 三表）
│   ├── models/           # schemas.py（请求数据契约）
│   ├── routers/          # chat.py（聊天 / 会话）· knowledge.py（知识库）
│   └── services/
│       ├── parsing/      # 文档解析层：base / pdf / docx / markdown / plain
│       ├── chunking.py   # 章节感知切分
│       ├── documents.py  # 文档入库 / 查询 / 删除（表与向量库同步）
│       ├── vector_store.py   # FAISS 懒加载与落盘
│       ├── embeddings.py # 本地 embedding 模型
│       ├── rag.py        # 检索与 prompt 组装
│       ├── llm.py        # DeepSeek 客户端
│       └── chat_history.py   # 会话与消息
├── requirements.txt
├── .env                  # 环境变量（本地创建，不入库）
├── chat.db               # 聊天与文档数据库（运行时生成）
└── faiss_store/          # 知识库向量索引（运行时生成）
```

## 快速开始

### 环境要求

- Python 3.10+
- 一个 DeepSeek API Key（[https://platform.deepseek.com](https://platform.deepseek.com) 申请）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> **依赖清单变动后要重装**：每次更新代码后，若 `requirements.txt` 有变化，请重新执行本步。否则新引入的包不会被安装，启动时会报 `ModuleNotFoundError`——例如缺 `pdfplumber` 会让后端**直接起不来**，前端表现为「连不上后端」。
>
> **确认装到了正确的 Python**：机器上若存在多个 Python（如 Anaconda 与独立 Python 并存），直接敲 `pip` 可能装到非目标环境，出现「明明装了却还缺包」。先用下面命令确认解释器位置，必要时改用完整路径安装：
>
> ```bash
> python -c "import sys; print(sys.executable)"                # 查看当前 pip 对应的解释器
> "C:\路径\python.exe" -m pip install -r requirements.txt       # 指定环境安装
> ```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入自己的密钥（**等号后不要留空格**）：

```
DEEPSEEK_API_KEY=sk-你的DeepSeek密钥
```

> 旧变量名 `dp_api_key` 仍然兼容，但建议统一使用 `DEEPSEEK_API_KEY`。

### 3. 启动项目

需要两个终端，先启动后端，再启动前端：

```bash
# 终端 1：后端
uvicorn backend:app --port 8000
```

```bash
# 终端 2：前端
streamlit run frontend.py
```

浏览器打开 Streamlit 提示的地址（默认 `http://localhost:8501`）即可使用。

> 首次上传文档时会加载本地 embedding 模型（约 10~15 秒），请耐心等待；之后会复用缓存。

## 接口一览

后端默认跑在 `http://127.0.0.1:8000`。

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | 流式问答（SSE） |
| GET | `/api/history/{session_id}` | 拉取某会话的历史消息 |
| GET | `/api/sessions` | 会话列表 |
| DELETE | `/api/sessions/{session_id}` | 删除会话 |
| POST | `/api/knowledge/upload` | 上传纯文本（JSON：`title` + `content`） |
| POST | `/api/knowledge/upload-file` | 上传文件（multipart，支持 PDF / docx / md / txt） |
| GET | `/api/knowledge` | 文档列表 |
| GET | `/api/knowledge/{doc_id}` | 文档详情（含分块与页码 / 章节） |
| DELETE | `/api/knowledge/{doc_id}` | 删除文档（同步清理向量库） |

## 使用方法

1. **聊天**：左侧选择或新建会话，直接输入问题，回复流式显示
2. **知识库**：点击「📚 知识库」面板 → 选择文件上传；上传后可在列表中查看文档，点 🔍 预览分块（含页码与章节），点 🗑 删除

## 常见问题

| 问题 | 解决 |
|---|---|
| 启动报 `ModuleNotFoundError` | 依赖没装齐（常见于代码更新后 `requirements.txt` 新增了包）。重新执行 `pip install -r requirements.txt`，并确认装到与启动服务相同的 Python 环境 |
| 首次上传很慢 | 首次要加载本地 embedding 模型（约 10~15 秒），属正常现象，之后会复用缓存 |
| 模型下载超时 | 代码已内置 `HF_ENDPOINT=https://hf-mirror.com`（见 `app/__init__.py`），若仍失败请检查网络 |
| 上传 `.doc` / `.wps` 报错 | 老版格式不支持，请在 Word 里另存为 `.docx` 或导出 PDF |
| PDF 上传后没有内容 | 该 PDF 可能是扫描件（没有文字层），会被标记为 `needs_ocr`，当前版本未接入 OCR |
| 鉴权失败 / 401 | 检查 `.env` 中 `DEEPSEEK_API_KEY` 是否正确，且等号后无空格 |
| 后端启动报端口占用 | 换端口启动：`uvicorn backend:app --port 8001`，并同步修改 `frontend.py` 里的接口地址 |

## 说明

- 本项目为个人学习项目，`.env`、`chat.db`、`faiss_store/` 等文件不参与版本管理
- 如需部署上线，请将密钥配置到服务端环境变量，并妥善处理 CORS 与安全策略
