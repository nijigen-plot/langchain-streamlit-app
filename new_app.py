import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder, PromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

# Agentを使う
def create_agent_chain():
    chat = ChatOpenAI(
            model_name=os.environ["OPENAI_API_MODEL"],
            temperature=os.environ["OPENAI_API_TEMPERATURE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            streaming=True
        )

    tools = load_tools(["ddg-search"], llm=chat)
    return initialize_agent(
        tools, 
        chat, 
        agent=AgentType.OPENAI_FUNCTIONS
    )

st.title('プラグインおすすめBOT')

# 動いてる途中に変更すると強制的に実行が上書きされるから後で変更したい
option = st.selectbox(
    '気になるFXプラグインは？',
    ('コンプレッサー','シンセサイザー','ディストーション','Colour Bassを作る時のおススメプラグイン')
)

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("あなたはDTM(Computer Music)における{option}のプロフェッショナルです。"),
    HumanMessagePromptTemplate.from_template("DTM(Computer Music)用のおすすめプラグインについて教えてください。\n\n プラグイン種類: {option}")
])

if st.button("実行", type="primary"):
    with st.chat_message("user"):
        st.markdown(option)
        
    # streamlitでストリーミングにするには工夫が必要っぽい
    # https://qiita.com/suzuki_sh/items/64d84c417cba43cd6351
    with st.chat_message("assistant"):
        # chat = ChatOpenAI(
        #     model_name=os.environ["OPENAI_API_MODEL"],
        #     temperature=os.environ["OPENAI_API_TEMPERATURE"],
        #     openai_api_key=os.environ["OPENAI_API_KEY"],
        #     streaming=True
        # )
        callback = StreamlitCallbackHandler(st.container())
        messages = chat_prompt.format_prompt(option=option).to_messages()
        response = create_agent_chain().invoke(messages, callback=[callback])
        # response = chat.invoke(messages)
        st.markdown(response['output'])
