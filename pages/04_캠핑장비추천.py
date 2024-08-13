import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# API KEY 정보로드
# load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("캠핑장비 추천 챗봇 🏕️")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)

    # 세션 ID 를 지정하는 메뉴
    session_id = st.text_input("세션 ID를 입력하세요.", "abc123")

# 캠핑장비 추천 로직
def recommend_gear(season, weather, people_count):
    base_gear = {
        '맑음 ☀️': '캠핑용 텐트 🏕️',
        '비 🌧️': '비옷 🌂, 방수 텐트 🛖, 우산 ☂️',
        '눈 ❄️': '눈 차단 장비 🧤, 방한복 🧥',
        '구름 많음 ☁️': '경량 텐트 ⛺, 다용도 방수 덮개 🛡️',
        '바람이 많이 불어요 🌬️': '튼튼한 텐트 🏕️, 바람 차단기 🌫️'
    }

    season_gear = {
        '봄 🌸': {'맑음 ☀️': '해먹 🪢, 선크림 ☀️', '비 🌧️': '', '눈 ❄️': '', '구름 많음 ☁️': '', '바람이 많이 불어요 🌬️': ''},
        '여름 🌞': {'맑음 ☀️': '햇볕 차단제 ☀️, 냉각 타올 🧊', '비 🌧️': '', '눈 ❄️': '', '구름 많음 ☁️': '', '바람이 많이 불어요 🌬️': ''},
        '가을 🍁': {'맑음 ☀️': '해먹 🪢, 따뜻한 옷 🧥', '비 🌧️': '', '눈 ❄️': '', '구름 많음 ☁️': '', '바람이 많이 불어요 🌬️': ''},
        '겨울 ❄️': {'맑음 ☀️': '방한복 🧥, 따뜻한 침낭 🛏️, 캠핑용 난로 🔥', '비 🌧️': '방한복 🧥, 우산 ☂️', '눈 ❄️': '스노우 보트 ⛷️', '구름 많음 ☁️': '방한복 🧥', '바람이 많이 불어요 🌬️': '방한복 🧥'}
    }

    gear = base_gear.get(weather, '').split(', ')
    additional_gear = season_gear.get(season, {}).get(weather, '').split(', ')

    if people_count == '2인 👫':
        gear += ['2인용 침낭 🛏️', '2인용 캠핑 의자 🪑']
    elif people_count == '4인 👨‍👩‍👧‍👦':
        gear += ['4인용 침낭 🛏️', '4인용 캠핑 의자 🪑']

    gear += additional_gear
    return gear

# 장비 리스트 생성
gear_list = []

# 체크박스 형태로 장비 리스트 출력
def display_checklist(gear_list):
    checked_items = []
    for item in gear_list:
        checked = st.checkbox(item)
        if checked:
            checked_items.append(item)

    if st.button('확인하기'):
        if checked_items:
            st.write('확인한 장비 목록:')
            for item in checked_items:
                st.write(f'- {item}')
        else:
            st.write('아직 체크한 장비가 없습니다.')

# 체인 생성
def create_chain(model_name="gpt-4o"):

    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 캠핑 장비 추천 챗봇입니다. 주어진 정보에 기반하여 적절한 캠핑 장비를 추천해 주세요.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#정보:\n{question}"),  # 사용자 입력을 변수로 사용
        ]
    )

    # llm 생성
    llm = ChatOpenAI(model_name=model_name, openai_api_key=st.session_state.get('api_key', ''))

    # 일반 Chain 생성
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return chain_with_history

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["store"] = {}

# 이전 대화 기록 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

if "multiturn_chain" not in st.session_state:
    st.session_state["multiturn_chain"] = create_chain(model_name=selected_model)

# 만약에 사용자 입력이 들어오면...
if user_input:
    chain = st.session_state["multiturn_chain"]
    if chain is not None:
        response = chain.stream(
            # 질문 입력
            {"question": user_input},
            # 세션 ID 기준으로 대화를 기록합니다.
            config={"configurable": {"session_id": session_id}},
        )

        # 사용자의 입력
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

            # 대화기록을 저장한다.
            add_message("user", user_input)
            add_message("assistant", ai_answer)

            # 캠핑 장비 추천을 위한 체크리스트 생성
            if "캠핑" in user_input:
                season = st.selectbox('계절을 선택해주세요:', ['봄 🌸', '여름 🌞', '가을 🍁', '겨울 ❄️'])
                weather = st.selectbox('날씨를 선택해주세요:', ['맑음 ☀️', '비 🌧️', '눈 ❄️', '구름 많음 ☁️', '바람이 많이 불어요 🌬️'])
                people_count = st.selectbox('캠핑에 참여하는 인원수를 선택해주세요:', ['솔로캠핑 👤', '2인 👫', '4인 👨‍👩‍👧‍👦'])

                gear_list = recommend_gear(season, weather, people_count)
                display_checklist(gear_list)
    else:
        # 이미지를 업로드 하라는 경고 메시지 출력
        warning_msg.error("이미지를 업로드 해주세요.")
