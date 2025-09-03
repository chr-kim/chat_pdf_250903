# 1. 필요한 라이브러리 import
import streamlit as st
import os
import tempfile
# from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from operator import itemgetter
from typing import List, Dict, Any

# 2. 페이지 설정
st.set_page_config(
    page_title="PDF RAG 챗봇 (LCEL)",
    page_icon="📚",
    layout="wide"
)

# 3. 환경 변수 및 API 키 설정
#@st.cache_data
#def load_environment():
    """환경변수 로드"""
#    load_dotenv()
#    return os.getenv("OPENAI_API_KEY")

# 4. PDF 처리 함수들 (기존과 동일)
@st.cache_data
def load_pdf(file_path):
    """PDF 로드 및 분할"""
    try:
        loader = PyPDFLoader(file_path)
        data = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        documents = text_splitter.split_documents(data)
        
        return documents, len(data)
    except Exception as e:
        st.error(f"PDF 로드 중 오류 발생: {str(e)}")
        return None, 0

@st.cache_resource
def create_vectorstore(documents, api_key):
    """벡터스토어 생성"""
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=api_key
        )
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"벡터스토어 생성 중 오류 발생: {str(e)}")
        return None

# 5. LCEL 기반 대화 체인 생성
def create_lcel_conversation_chain(vectorstore, api_key):
    """LCEL을 사용한 대화형 RAG 체인 생성"""
    try:
        # LLM 초기화
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=api_key,
            temperature=0
        )
        
        # 리트리버 생성
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # 질문 재구성 프롬프트 (대화 히스토리 고려)
        contextualize_q_system_prompt = (
            "주어진 채팅 기록과 최신 사용자 질문이 있을 때, "
            "채팅 기록의 맥락을 참조할 수 있는 질문을 "
            "채팅 기록 없이도 이해할 수 있는 독립적인 질문으로 재구성하세요. "
            "질문에 답하지 말고, 필요한 경우에만 재구성하고 "
            "그렇지 않으면 그대로 반환하세요."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # 질문 재구성 체인
        history_aware_retriever = contextualize_q_prompt | llm | StrOutputParser() | retriever
        
        # QA 프롬프트
        qa_system_prompt = (
            "당신은 질문 답변 작업을 위한 어시스턴트입니다. "
            "다음의 검색된 맥락을 사용하여 질문에 답하세요. "
            "답을 모르면 모른다고 말하세요. "
            "답변은 최대 세 문장으로 간결하게 작성하세요. "
            "한국어로 답변해주세요.\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # 문서 결합 체인
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # 최종 RAG 체인
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain
        
    except Exception as e:
        st.error(f"LCEL 체인 생성 중 오류 발생: {str(e)}")
        return None

# 6. 채팅 히스토리 관리 함수들
def format_chat_history(chat_history: List[tuple]) -> List:
    """채팅 히스토리를 LangChain 메시지 형식으로 변환"""
    formatted_history = []
    for human_msg, ai_msg in chat_history:
        formatted_history.append(HumanMessage(content=human_msg))
        formatted_history.append(AIMessage(content=ai_msg))
    return formatted_history

def get_session_history() -> List:
    """세션에서 채팅 히스토리 가져오기"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    return format_chat_history(st.session_state.chat_history)

# 7. 메인 애플리케이션
def main():
    st.title("📚 PDF RAG 챗봇 (LCEL 버전)")
    st.markdown("**LCEL(LangChain Expression Language)**을 활용한 현대적 RAG 챗봇")
    
    # API 키 확인
    api_key = load_environment()
    if not api_key:
        st.warning("⚠️ OpenAI API 키를 설정해주세요.")
        api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
        if not api_key:
            st.stop()
    
    # 사이드바 - 파일 업로드 및 설정
    with st.sidebar:
        st.header("📁 문서 업로드")
        
        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type="pdf",
            help="PDF 파일만 업로드 가능합니다."
        )
        
        use_default = st.checkbox("기본 파일 사용 (소나기.pdf)")
        
        if st.button("🔄 문서 처리 시작"):
            if uploaded_file or use_default:
                with st.spinner("📖 문서를 처리하고 있습니다..."):
                    # 파일 경로 설정
                    if use_default:
                        file_path = './data/소나기.pdf'
                        if not os.path.exists(file_path):
                            st.error("기본 파일이 존재하지 않습니다. 파일을 업로드해주세요.")
                            return
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            file_path = tmp_file.name
                    
                    # 문서 처리
                    documents, page_count = load_pdf(file_path)
                    
                    if documents:
                        st.session_state.documents = documents
                        st.session_state.page_count = page_count
                        
                        # 벡터스토어 생성
                        vectorstore = create_vectorstore(documents, api_key)
                        
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            
                            # LCEL 체인 생성
                            rag_chain = create_lcel_conversation_chain(vectorstore, api_key)
                            
                            if rag_chain:
                                st.session_state.rag_chain = rag_chain
                                st.success(f"✅ LCEL 체인 생성 완료! ({page_count}개 페이지, {len(documents)}개 청크)")
                    
                    # 임시 파일 정리
                    if not use_default and os.path.exists(file_path):
                        os.unlink(file_path)
            else:
                st.warning("파일을 업로드하거나 기본 파일 사용을 선택해주세요.")
        
        # 체인 정보 표시
        if 'rag_chain' in st.session_state:
            st.subheader("🔗 체인 정보")
            st.success("LCEL RAG 체인 활성화")
            
            with st.expander("📋 기술적 세부사항"):
                st.markdown("""
                **사용된 LCEL 컴포넌트:**
                - `ChatPromptTemplate`: 프롬프트 템플릿
                - `create_retrieval_chain`: 검색 체인
                - `create_stuff_documents_chain`: 문서 결합 체인
                - `MessagesPlaceholder`: 채팅 히스토리 관리
                - History-aware retriever로 대화 맥락 고려
                """)
    
    # 메인 영역 - 채팅 인터페이스
    if 'rag_chain' in st.session_state:
        st.header("💬 LCEL 기반 채팅")
        
        # 채팅 기록 초기화
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # 채팅 기록 표시
        chat_container = st.container()
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f"**🧑‍💻 질문:** {question}")
                st.markdown(f"**🤖 답변:** {answer}")
                st.divider()
        
        # 질문 입력
        question = st.text_input(
            "질문을 입력하세요:",
            placeholder="예: 소년은 어디에서 처음 소녀를 만났나요?",
            key="question_input"
        )
        
        # 질문 처리 (LCEL 체인 사용)
        if st.button("📤 질문하기") and question:
            with st.spinner("🤔 LCEL 체인으로 답변 생성 중..."):
                try:
                    # 현재 채팅 히스토리 가져오기
                    chat_history = get_session_history()
                    
                    # LCEL 체인 실행
                    result = st.session_state.rag_chain.invoke({
                        "input": question,
                        "chat_history": chat_history
                    })
                    
                    answer = result["answer"]
                    
                    # 채팅 기록에 추가
                    st.session_state.chat_history.append((question, answer))
                    
                    # 검색된 문서 정보 표시 (선택사항)
                    if st.checkbox("🔍 검색된 문서 정보 보기"):
                        with st.expander("📄 검색된 문서들"):
                            for i, doc in enumerate(result.get("context", [])):
                                st.write(f"**문서 {i+1}:**")
                                st.write(doc.page_content[:300] + "...")
                                st.divider()
                    
                    # 페이지 새로고침
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"LCEL 체인 실행 중 오류 발생: {str(e)}")
        
        # 채팅 기록 초기화 버튼
        if st.session_state.chat_history:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🗑️ 채팅 기록 초기화"):
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                if st.button("💾 대화 내보내기"):
                    conversation_text = ""
                    for q, a in st.session_state.chat_history:
                        conversation_text += f"Q: {q}\nA: {a}\n\n"
                    
                    st.download_button(
                        label="📥 대화 다운로드",
                        data=conversation_text,
                        file_name="conversation_history.txt",
                        mime="text/plain"
                    )
        
        # 문서 정보 표시
        with st.expander("📊 문서 및 체인 정보"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("페이지 수", st.session_state.page_count)
                st.metric("청크 수", len(st.session_state.documents))
            with col2:
                st.metric("대화 수", len(st.session_state.chat_history))
                st.write("**체인 타입:** LCEL RAG Chain")
    
    else:
        st.info("👆 사이드바에서 PDF 문서를 업로드하고 처리를 시작해주세요.")
        
        # LCEL 장점 설명
        st.subheader("🚀 LCEL의 장점")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔧 기술적 개선사항:**
            - 더 명확한 체인 구조
            - 스트리밍 지원
            - 비동기 처리 지원
            - 더 나은 오류 처리
            """)
        
        with col2:
            st.markdown("""
            **💡 기능적 개선사항:**
            - 대화 맥락 인식
            - 질문 재구성 기능
            - 검색 문서 반환
            - 모듈화된 구조
            """)
        
        # 예시 질문 표시
        st.subheader("💡 예시 질문들")
        example_questions = [
            "소년은 어디에서 처음 소녀를 만났나?",
            "소년은 소녀에게 무엇을 주려고 했나?", 
            "그 장소는 어떤 특징이 있었나?",
            "소나기는 언제 내렸나?"
        ]
        
        for q in example_questions:
            st.markdown(f"• {q}")

# 8. 애플리케이션 실행
if __name__ == "__main__":
    main()
