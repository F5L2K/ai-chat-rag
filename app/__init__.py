# HuggingFace 国内镜像必须在 sentence_transformers 被导入前设置，
# 否则模型下载会直连 huggingface.co 导致超时。
# 放在包初始化里最保险：任何 app.* 的导入都会先执行这里。
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
