"""DeepSeek 客户端。全项目共用一个实例。"""

from openai import OpenAI

from app.core.config import API_KEY, DEEPSEEK_BASE_URL

client = OpenAI(api_key=API_KEY, base_url=DEEPSEEK_BASE_URL)
