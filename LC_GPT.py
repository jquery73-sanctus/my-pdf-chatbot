import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
# 에러 방지를 위해 경로를 최신 버전으로 지정
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.callbacks import get_openai_callback

# 1. GUI 화면 설정
st.set_page_config(page_title="PDF 가이드 챗봇", layout="wide")
st.title("📄 PDF에게 무엇이든 물어보세요 (토큰 추적 포함)")

# 사이드바에서 설정값 입력 받기
with st.sidebar:
    st.header("설정")
    openai_key = st.text_input("OpenAI API Key", type="password")
    st.info("API Key를 입력하고 PDF를 업로드하면 대화가 시작됩니다.")

# 2. PDF 업로드 및 처리 로직
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

# 세션 상태 초기화 (채팅 기록 저장용)
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file and openai_key:
    # 임시 파일 저장 (PyPDFLoader는 파일 경로가 필요함)
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # [과정 1] 문서 로드 및 쪼개기
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    # [과정 2] 벡터 저장소 만들기 (FAISS)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # [과정 3] 랭체인 연결 (QA Chain)
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_key, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    st.success("✅ PDF 분석 완료! 이제 대화를 시작하세요.")

    # 3. 채팅 UI 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 질문 입력
    if prompt := st.chat_input("문서 내용에 대해 질문해주세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 답변 생성 및 토큰 추적
        with st.chat_message("assistant"):
            with get_openai_callback() as cb:
                # 최신 권장 방식인 invoke 사용
                result = qa_chain.invoke(prompt)
                response = result['result'] # 결과 딕셔너리에서 답변만 추출
                
                st.markdown(response)
                
                # 토큰 사용량 및 비용 정보 출력
                st.info(f"""
                **💰 이번 질문의 비용 영수증:**
                - 사용된 총 토큰: {cb.total_tokens}
                - 상세: (입력 {cb.prompt_tokens} / 출력 {cb.completion_tokens})
                - 예상 비용: ${cb.total_cost:.5f} (약 {cb.total_cost * 1400:.2f}원)
                """)
                
                st.session_state.messages.append({"role": "assistant", "content": response})

elif not openai_key:
    st.warning("👈 왼쪽 사이드바에 OpenAI API Key를 입력해 주세요.")
