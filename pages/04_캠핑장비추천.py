import streamlit as st
import requests
from PIL import Image
from io import BytesIO

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("캠핑 장비 추천 챗봇 🏕️")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}

if "image_analysis" not in st.session_state:
    st.session_state["image_analysis"] = ""

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)

    # 세션 ID 를 지정하는 메뉴
    session_id = st.text_input("세션 ID를 입력하세요.", "abc123")


# 이미지 분석 함수 (이 예제에서는 OpenAI의 이미지 모델을 사용한다고 가정)
def analyze_image(image):
    # 이 부분은 실제 이미지 분석을 위한 API 호출이나 모델 실행으로 대체해야 합니다.
    # 여기서는 예시로 단순히 예측 결과를 반환합니다.
    return "산악 지형, 맑은 날씨"

# 장비 추천 함수
def recommend_equipment(description):
    # 장비 추천을 위한 간단한 로직 예제
    if "산악" in description and "맑은" in description:
        return "추천 장비: 등산용 텐트, 트레킹 폴, 다용도 나이프"
    else:
        return "추천 장비: 일반 캠핑 장비"

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]


# 체인 생성
def create_chain(model_name="gpt-4o"):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 캠핑 장비 추천 챗봇입니다."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Location and Weather Description:\n{description}"),
        ]
    )

    llm = ChatOpenAI(model_name=model_name, openai_api_key=st.session_state.get('api_key', ''))
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="description",
        history_messages_key="chat_history",
    )
    return chain_with_history


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["image_analysis"] = ""

# 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    # 이미지 분석
    analysis_result = analyze_image(image)
    st.session_state["image_analysis"] = analysis_result
    st.write(f"분석된 장소 및 날씨: {analysis_result}")

    # 장비 추천
    recommendation = recommend_equipment(analysis_result)
    st.write(f"추천 장비: {recommendation}")

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if "multiturn_chain" not in st.session_state:
    st.session_state["multiturn_chain"] = create_chain(model_name=selected_model)

if user_input:
    chain = st.session_state["multiturn_chain"]
    if chain is not None:
        response = chain.stream(
            {"description": st.session_state["image_analysis"]},
            config={"configurable": {"session_id": session_id}},
        )

        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

            add_message("user", user_input)
            add_message("assistant", ai_answer)
    else:
        st.error("체인을 초기화해 주세요.")
