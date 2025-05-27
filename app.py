import streamlit as st
from boj_llmrec import LLMRec

api_key = st.text_input("🔐 OpenAI API 키를 입력하세요", type="password")

# 세션 상태 초기화
if 'session' not in st.session_state:
    st.session_state.session = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# LLMRec 로드
@st.cache_resource
def load_llmrec(api_key):
    return LLMRec(api_key=api_key)

# Streamlit UI
st.title("🧠 LLM 기반 알고리즘 문제 추천 챗봇")
st.markdown("사용자 핸들을 입력하면 추천 문제 기반으로 대화가 가능합니다.")

if api_key:
    llmrec = load_llmrec(api_key)

    user_handle = st.text_input("🔍 사용자 핸들 입력", value="37aster")

    if st.button("새로운 세션 시작"):
        st.session_state.session = llmrec.get_new_session(user_handle)
        st.session_state.chat_history = []
        st.success("세션이 시작되었습니다. 메시지를 입력해보세요!")

    if st.session_state.session:
        message = st.text_input("💬 메시지 입력", key="user_msg")
        if st.button("전송") and message.strip():
            with st.spinner("LLM 응답 생성 중..."):
                text_response, speech_response = st.session_state.session.chat(message)
                st.session_state.chat_history.append(("👤", message))
                st.session_state.chat_history.append(("🤖", text_response))
                st.session_state.chat_history.append(("🔊", speech_response))
                st.success(f"세션 제목: {st.session_state.session.title}")

    if st.session_state.chat_history:
        st.markdown("---")
        for speaker, msg in st.session_state.chat_history:
            st.markdown(f"**{speaker}**: {msg}")
else:
    st.warning("먼저 OpenAI API 키를 입력하세요.")
