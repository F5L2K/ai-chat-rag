# AIChat - 带知识库的 AI 聊天助手

基于 **FastAPI + Streamlit + DeepSeek + RAG** 的本地 AI 聊天助手。支持多会话聊天、流式输出，以及上传文档构建本地知识库（向量检索问答）。

## 功能特性

- 💬 **多会话聊天**：会话自动保存到本地 SQLite，刷新页面不丢失，支持新建 / 切换 / 删除会话
- ⚡ **流式输出**：AI 回复逐字显示，无需等待完整生成
- 📚 **RAG 知识库**：上传文档后自动切块、向量化并建立索引，提问时检索最相关片段辅助回答
- 🧠 **本地向量化**：使用 `BAAI/bge-small-zh-v1.5` 本地 embedding 模型，不依赖额外 API
- 🇨🇳 **国内镜像**：模型下载自动走 HuggingFace 国内镜像（`hf-mirror.com`），无需科学上网

## 技术栈

| 部分 | 技术 |
|---|---|
| 后端 | Python · FastAPI · Uvicorn |
| 前端 | Streamlit |
| 大模型 | DeepSeek（OpenAI 兼容接口） |
| 向量检索 | LangChain · FAISS · sentence-transformers |
| 存储 | SQLite（聊天记录）· 本地向量索引（faiss_store/） |

## 目录结构

```
AIChat/
├── backend.py        # FastAPI 后端：聊天、会话、知识库 API
├── frontend.py       # Streamlit 前端界面
├── main.py           # PyCharm 模板文件（非入口，可忽略）
├── requirements.txt  # 依赖清单
├── .env              # 环境变量（本地创建，不入库）
├── chat.db           # 聊天记录数据库（运行时生成）
└── faiss_store/      # 知识库向量索引（运行时生成）
```

## 快速开始

### 环境要求

- Python 3.10+
- 一个 DeepSeek API Key（[https://platform.deepseek.com](https://platform.deepseek.com) 申请）

### 1. 克隆项目

```bash
git clone https://github.com/F5L2K/ai-chat-rag.git
cd ai-chat-rag
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录新建 `.env` 文件，填入你自己的密钥（**等号后不要留空格**）：

```
dp_api_key=sk-你的DeepSeek密钥
```

> 首次运行会自动下载中文 embedding 模型（约 100MB，走国内镜像），请保持网络畅通。

### 4. 启动项目

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

## 使用方法

1. **聊天**：左侧选择或新建会话，直接输入问题，回复流式显示
2. **知识库**：点击「📚 知识库」面板，上传文档（txt / md / pdf 等），自动建立索引后即可基于文档内容提问

## 常见问题

| 问题 | 解决 |
|---|---|
| 模型下载超时 | 代码已内置 `HF_ENDPOINT=https://hf-mirror.com`，若仍失败请检查网络 |
| 鉴权失败 / 401 | 检查 `.env` 中 `dp_api_key` 是否正确，且等号后无空格 |
| 后端启动报端口占用 | 换端口启动：`uvicorn backend:app --port 8001`，并同步修改 `frontend.py` 里的接口地址 |

## 说明

- 本项目为个人学习项目，`.env`、`chat.db`、`faiss_store/` 等文件不参与版本管理
- 如需部署上线，请将密钥配置到服务端环境变量，并妥善处理 CORS 与安全策略
