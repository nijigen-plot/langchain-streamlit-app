import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

st.title('DTMプラグイン おすすめBOT')

# チャットBOT応答中はセレクトBOXを使わせないようにする
if "execute" not in st.session_state:
    st.session_state.disabled = False
else:
    st.session_state.disabled = True
    
# 動いてる途中に変更すると強制的に実行が上書きされるから後で変更したい->実行中は弄れないようにすればええやんけ
option = st.selectbox(
    '気になるプラグインは？',
    (
        'Compressor',
        'シンセサイザー',
        'ディストーション',
        'Colour Bassを作る時のおススメプラグイン'
    ),
    disabled=st.session_state.disabled,
    key='question'
)

# チャットBOT応答後、セレクトBOXのDisabledが解除されないため、再読み込みして解除する。
# その際、応答したメッセージが消えてしまうので残すように処置
if "messages" not in st.session_state:
    st.session_state.messages = []

# Stream描画の為のクラス
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Agentを使う
def create_agent_chain():
    container = st.empty()
    stream_handler = StreamHandler(container)
    chat = ChatOpenAI(
            model_name=os.environ["OPENAI_API_MODEL"],
            temperature=os.environ["OPENAI_API_TEMPERATURE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            streaming=True,
            callbacks=[stream_handler]
        )

    tools = load_tools(["ddg-search"], llm=chat)
    return initialize_agent(
        tools, 
        chat, 
        agent=AgentType.OPENAI_FUNCTIONS
    )

# プロンプト設定
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("あなたはDTM(Computer Music)における{option}のプロフェッショナルです。"),
    HumanMessagePromptTemplate.from_template("DTM(Computer Music)用のおすすめプラグインについて教えてください。\n\n プラグイン種類: {option}")
])

if st.button("実行", type="primary", key="execute"):
    # ユーザーの選択内容をst.session_satte.messagesに追加
    st.session_state.messages.append({"role":"user", "content":option})
    with st.chat_message("user"):
        st.markdown(option)
        
    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        messages = chat_prompt.format_prompt(option=option).to_messages()
        response = create_agent_chain().invoke(messages, callback=[callback])
        # 応答をst.session_state.messagesに追加
        st.session_state.messages.append({"role":"assistant", "content": response["output"]})
        del st.session_state["execute"] # 応答後、execute session_stateを消してSelectboxを有効化
        st.rerun() # もう１回上から実行させる何かしらが必要。rerunはテキストもすべて消えるので極端だが、状態を維持して上から再実行するみたいなのがあれば好ましい 参考書の出力履歴を残すやつが使えるかもしれない

# rerun実行後に出力を表示する。一番最後に置かないとSelecltboxやbuttonより上に来てしまう。
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

del st.session_state["messages"]    
del st.session_state["execute"]