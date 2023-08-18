import threading
from flask import Flask, request, jsonify
from datetime import datetime
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
import requests
import pytz

# Flask 부분

app = Flask(__name__)

API_CALL_LIMIT = 10
ip_data = {}
korea_timezone = pytz.timezone('Asia/Seoul')

@app.route('/api/call', methods=['POST'])
def api_call():
    ip = request.remote_addr
    # 현재 시간을 한국 시간대로 가져옴
    current_time = datetime.now(korea_timezone)
    current_date = current_time.date()
    # IP가 기록되지 않았을 때 초기화
    if ip not in ip_data:
        ip_data[ip] = {'count': 0, 'last_date': None}

    # 현재 날짜가 마지막 호출일과 다를 경우 초기화
    if ip_data[ip]['last_date'] != current_date:
        ip_data[ip]['count'] = 0
        ip_data[ip]['last_date'] = current_date

    # API 호출 횟수 제한 확인
    if ip_data[ip]['count'] < API_CALL_LIMIT:
        ip_data[ip]['count'] += 1  # 호출 횟수 증가
        return jsonify(success=True, count=ip_data[ip]['count'])
    else:
        return jsonify(success=False, message="하루 API 호출 횟수를 초과했습니다.")

def run_flask():
    app.run(port=5001)  # 포트 번호를 5001로 변경

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

# Flask 서버의 공용 URL을 설정합니다. 위의 셀에서 출력된 URL로 변경해야 한다
FLASK_SERVER_URL = "http://localhost:5001"  # 예: "http://12345678.ngrok.io"

# API 키 설정 및 초기화
os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]

vectordb = Chroma(persist_directory='./db', embedding_function=OpenAIEmbeddings())
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
qa_chain_global = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model='gpt-3.5-turbo', temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

st.title(':mag_right: Streamlit QA챗봇')
chat_history = []
user_input = st.text_input(label=':seedling: 당신이 궁금한 것을 질문해보세요!', placeholder='QA검색')

# Send 버튼을 누르면 공지사항을 숨김
if st.button("Send"):
    st.session_state.show_notice = False  

if 'show_notice' not in st.session_state or st.session_state.show_notice:
    st.write("📢 2017학년도 ~ 2024학년도 평가원 비문학 지문에 묻고 답하는 챗봇입니다")
    st.write("📌 구체적으로 물어볼수록 더 자세하게 답합니다")
    st.write("📌 방대한 비문학 문서에서 스스로 찾아서 대답합니다!")
    st.session_state.show_notice = True

if not st.session_state.show_notice:
    with st.spinner("Waiting for QA챗봇..."):
        if user_input:
            response = requests.post(FLASK_SERVER_URL + "/api/call")
            data = response.json()

            if data["success"]:
                chat_history.append({"user": user_input})
                llm_response = qa_chain_global(user_input + " 두문장으로 간단하게 ~다로 문장을 무조건적으로 끝맺어줘")
                bot_response = llm_response['result']
                chat_history.append({"bot": bot_response})

                for message in chat_history:
                    if "user" in message:
                        st.write(":green[You]: ", message["user"])
                    else:
                        st.write(":blue[Bot]: ", message["bot"])
                        st.write('\n\nSources:')
                        for source in llm_response["source_documents"]:
                            st.write(source.metadata['source'])
            else:
                st.warning(f"API 호출 제한을 초과했습니다! 하루에 {API_CALL_LIMIT}번만 API를 호출할 수 있습니다.")
