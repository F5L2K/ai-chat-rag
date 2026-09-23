import os                                  #Python 内置，读写环境变量
from dotenv import load_dotenv             #读取.env配置文件第三方库
from openai import OpenAI                  #兼容 OpenAI 协议客户端，deepseek、通义千问都兼容这套接口，所以用这个类
import streamlit as st                     #streamlit 全部功能都用st.xxx调用,具体功能详见P2

#加载.env
load_dotenv()

#初始化大模型客户端      所有发请求、收回复，全部通过这个 client 对象
client=OpenAI(
    api_key=os.getenv("dp_api_key"),      #身份凭证，证明你有权调用接口
    base_url="https://api.deepseek.com"   #接口地址，切换不同模型服务商只改这个地址 + model 名字
              )

# 初始化会话状态：如果messages不存在，就初始化空列表
if "messages" not in st.session_state:
    # 首次打开页面，初始化对话列表，第一条就是system提示词
    st.session_state.messages = [
        {"role":"system", "content":"你是严谨的数学助手，计算步骤写清楚，不要直接甩答案。"}
    ]



# 遍历历史消息，渲染到页面
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):   #with内部缩进下的所有代码，都包裹在这个聊天气泡框里面     打开盒子，缩进内的内容全部装进去；缩进结束，盒子关上
                                         # st.chat_message(role)：Streamlit 内置函数，用来创建一个聊天气泡容器
        st.markdown(msg["content"])      #在页面渲染文本，支持换行、标题、列表等 markdown 格式


# 聊天输入框
user_input = st.chat_input("请输入你的问题")#streamlit 内置聊天输入框

# 判断用户是否输入内容
if user_input:
    # 1.把用户消息加入会话状态
    st.session_state.messages.append({"role":"user", "content": user_input})
    # 2.页面渲染用户说的话
    with st.chat_message("user"):
        st.markdown(user_input)

    # ==========在这里调用大模型==========
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=st.session_state.messages,
        stream=False
    )
    ai_reply = resp.choices[0].message.content

    # 3.把AI回复存入历史
    st.session_state.messages.append({"role":"assistant", "content": ai_reply})
    # 4.渲染AI回复
    with st.chat_message("assistant"):
        st.markdown(ai_reply)

