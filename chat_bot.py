# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv("data/.env")
api_key = os.getenv("OPENAI_API_KEY")

# PDF 처리 함수
@st.cache_resource
def process_pdf():
    loader = PyPDFLoader("data/2024_KB_부동산_보고서_최종.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# 벡터 스토어 초기화
@st.cache_resource
def initialize_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    if os.path.exists("index.faiss"):
        return FAISS.load_local(".", embeddings, allow_dangerous_deserialization=True)
    else:
        chunks = process_pdf()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(".")
        return vectorstore

# 체인 초기화
def get_trimmed_history(session_id):
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = ChatMessageHistory()
    history = st.session_state.chat_histories[session_id]
    if len(history.messages) > 4:
        history.messages = history.messages[-4:]
    return history

def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """당신은 KB 부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.

    컨텍스트: {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        base_chain,
        lambda session_id: get_trimmed_history(session_id),
        input_messages_key="question",
        history_messages_key="chat_history",
    )
##############################################################
def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠 KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("부동산 관련 질문을 입력하세요"):
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 체인 초기화
        chain = initialize_chain()

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question": prompt},
                    {"configurable": {"session_id": "streamlit_session"}}
                )
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

from pyngrok import ngrok
NGROK_AUTH_TOKEN = "3CGZKTM3Wahcad3UP5tTcPWj1lP_2LriDU4DV75a7JjgkeEy4"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

public_url = ngrok.connect(8501)
print("앱 접속 URL:" ,public_url)
