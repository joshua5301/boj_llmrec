import streamlit as st
from boj_llmrec import LLMRec

st.title("🧠 LLM 기반 알고리즘 문제 추천 챗봇")
st.markdown("사용자 핸들과 프로필을 입력하면 맞춤형 문제 추천이 가능합니다.")

# 1. OpenAI API 키 입력
api_key = st.text_input("🔐 OpenAI API 키를 입력하세요", type="password")

# 2. 세션 상태 초기화
if 'session' not in st.session_state:
    st.session_state.session = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 3. LLMRec 로드
@st.cache_resource
def load_llmrec(api_key):
    return LLMRec(api_key=api_key)

# 4. 사용자 정보 입력
if api_key:
    llmrec = load_llmrec(api_key)
    st.subheader("👤 사용자 프로필 입력")

    user_handle = st.text_input("🔍 사용자 핸들 입력", value="37aster")

    user_level = st.selectbox("📊 현재 알고리즘 실력", options=[
        "very low", "low", "medium", "high", "very high"
    ])

    goal = st.selectbox("🎯 문제 풀이 목적", options=[
        "coding test", "contest", "learning", "hobby"
    ])

    interested_tags = st.multiselect(
        "🧩 흥미있는 알고리즘 주제 (복수 선택 가능)",
        options=["DP", "그래프", "자료구조", "수학", "구현", "문자열", "탐욕법", "트리", "정렬"]
    )

    # 5. 새로운 세션 시작
    if st.button("🚀 새로운 세션 시작"):
        profile = {
            "user_level": user_level,
            "goal": goal,
            "interested_tags": interested_tags,
        }
        st.session_state.session = llmrec.get_new_session(user_handle, profile=profile)
        st.session_state.chat_history = []
        st.success("세션이 시작되었습니다. 메시지를 입력해보세요!")

    # 6. 대화 입력 및 출력
    if st.session_state.session:
        message = st.text_input("💬 메시지 입력", key="user_msg")
        if st.button("전송") and message.strip():
            with st.spinner("LLM 응답 생성 중..."):
                text_response, speech_response, keywords = st.session_state.session.chat(message)
                st.session_state.chat_history.append(("👤", message))
                st.session_state.chat_history.append(("🤖", text_response))
                st.session_state.chat_history.append(("🔊", speech_response))
                st.session_state.chat_history.append(("🔑", keywords))
                st.success(f"세션 제목: {st.session_state.session.title}")

    # 7. 대화 히스토리 출력
    if st.session_state.chat_history:
        st.markdown("---")
        for speaker, msg in st.session_state.chat_history:
            st.markdown(f"**{speaker}**: {msg}")
else:
    st.warning("먼저 OpenAI API 키를 입력하세요.")
