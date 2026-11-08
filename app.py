"""Run: python -m streamlit run app.py"""
import time
import pandas as pd
import streamlit as st
from agent import answer
from config import settings
from exam_service import prepare_exam, fingerprint, excel_bytes

st.set_page_config(page_title='클래스인사이트 · 영어 평가 도우미',page_icon='📘',layout='wide')
st.markdown('''<style>
.block-container{max-width:1240px;padding-top:2rem}
.hero{padding:40px;border-radius:24px;background:linear-gradient(115deg,#113b46,#1b766c);color:white;margin:12px 0 28px}
.hero h1,.hero p{color:white}.eyebrow{font-size:12px;letter-spacing:3px;opacity:.75}
.hero h1{font-size:38px;line-height:1.3}.hero p{font-size:17px;max-width:690px}
[data-testid="stMetric"]{background:white;border:1px solid #e3e9ed;border-radius:16px;padding:18px}
.step{background:white;border:1px solid #e3e9ed;border-radius:16px;padding:24px;min-height:170px}
.step b{color:#176b61;font-size:20px}.step p{line-height:1.7;color:#526477}
</style>''',unsafe_allow_html=True)

def reset_analysis():
    for key in ['exam','excel','messages','memory','last_result','current_signature']:
        st.session_state.pop(key,None)

def home():
    import base64
    from pathlib import Path

    logo_path = Path(__file__).resolve().parent / "assets" / "gachon_logo.png"

    if not logo_path.is_file():
        st.error("assets 폴더에 gachon_logo.png 파일을 넣어주세요.")
        return

    logo_base64 = base64.b64encode(
        logo_path.read_bytes()
    ).decode("utf-8")

    st.markdown("""
    <style>
    .landing-main {
        min-height: 65vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 30px 16px 70px;
        text-align: center;
    }

    .landing-logo {
        width: 190px;
        max-width: 55vw;
        height: auto;
        margin-bottom: 42px;
    }

    .landing-title {
        font-size: clamp(24px, 4vw, 36px);
        font-weight: 700;
        color: #233f78;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        # 화면 왼쪽 위 페이지 이동 버튼
        if st.button("시험 분석 페이지 →", key="go_to_analysis"):
            st.switch_page(analysis_page)

        # 화면 중앙 로고와 서비스 이름
        st.markdown(
            f"""
            <div class="landing-main">
                <img
                    class="landing-logo"
                    src="data:image/png;base64,{logo_base64}"
                    alt="가천대학교 로고"
                >
                <h1 class="landing-title">영어 학원 평가 도우미</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

def workspace():
    st.title('시험 분석실')
    st.caption('채점 결과를 확인하고, 다음 수업에 필요한 질문을 해보세요.')
    with st.sidebar:
        st.subheader('시험 자료')
        qp=st.file_uploader('① 영어 문제지 PDF',type=['pdf'],key='question_upload',on_change=reset_analysis)
        ap=st.file_uploader('② 정답·배점 PDF',type=['pdf'],key='answer_upload',on_change=reset_analysis)
        sp=st.file_uploader('③ 학생 OMR 응답 엑셀',type=['xlsx'],key='student_upload',on_change=reset_analysis)
        ready=all(x is not None for x in [qp,ap,sp])
        files=[sp.getvalue(),ap.getvalue(),qp.getvalue()] if ready else None
        signature=fingerprint(files) if files else None
        if ready and st.session_state.get('exam') and signature!=st.session_state.get('current_signature'):
            reset_analysis();st.info('자료가 변경되어 이전 채점 결과와 대화를 초기화했습니다.')
        clicked=st.button('분석 시작',disabled=not ready,type='primary',width='stretch')
        if clicked:
            reset_analysis()
            with st.status('자료를 읽고 채점하고 있습니다…',expanded=True) as status:
                try:
                    st.write('정답·배점 검증 → 학생 답안 채점 → 문항별 지문과 이미지 연결')
                    started=time.monotonic();exam=prepare_exam(*files)
                    exported=excel_bytes(exam)
                    st.session_state.update(exam=exam,excel=exported,current_signature=signature,messages=[],memory={})
                    status.update(label=f'채점 완료 · {len(exam.students)}명 / {time.monotonic()-started:.1f}초',state='complete',expanded=False)
                except Exception as exc:
                    status.update(label='파일을 확인해주세요',state='error')
                    st.error('자료 구조를 확인하지 못했습니다. 문제지·정답표 유형, 문항 수, 학생명 중복과 답안 값을 확인해주세요.')
                    # Parser errors contain data, not credentials; useful to the uploader.
                    with st.expander('입력 오류 상세'):st.text(str(exc)[:600])
        exam=st.session_state.get('exam')
        st.download_button('엑셀 성적표 다운로드',data=st.session_state.get('excel',b''),
            file_name='영어시험_성적분석.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            disabled=exam is None,width='stretch')
        st.caption('파일을 바꾸면 새 시험으로 다시 분석합니다.')
    if exam is None:
        st.info('왼쪽에서 파일 3개를 올린 뒤 ‘분석 시작’을 눌러주세요.')
        st.chat_input('채점이 완료되면 질문할 수 있습니다.',disabled=True)
        return
    cols=st.columns(4)
    for col,label,value in zip(cols,['학생 수','평균','중앙값','최다 오답 문항'],[
        f'{len(exam.students)}명',f"{exam.stats['mean']:.2f}점",f"{exam.stats['median']:.2f}점",f"{exam.stats['item_errors'][0]['question_id']}번"]):col.metric(label,value)
    with st.expander('성적표와 문항별 근거 보기'):
        left,right=st.columns(2)
        left.dataframe(pd.DataFrame(exam.scores).rename(columns={'student_id':'학생명','score':'점수'}),hide_index=True)
        right.dataframe(pd.DataFrame(exam.stats['item_errors']).rename(columns={'question_id':'문항','wrong_count':'오답 수','wrong_rate':'오답률'}),hide_index=True)
        number=st.selectbox('원본 확인 문항',sorted(exam.key))
        record,images=exam.question_context(number)
        st.caption(f"정답 {record['answer']}번 · 배점 {record['points']}점 · {record['type']}")
        c1,c2=st.columns(2)
        for im in images:c1.image(im)
        c2.text(record['complete_context'])
    st.subheader('성적에 대해 물어보세요')
    buttons=st.columns(3);chosen=None
    for col,label in zip(buttons,['평균과 중앙값 알려줘','가장 성적이 저조한 학생은 누구야?','가장 많이 틀린 문제와 유사한 문제 1개 만들어줘']):
        if col.button(label,width='stretch'):chosen=label
    for message in st.session_state.get('messages',[]):
        with st.chat_message(message['role']):st.markdown(message['content'])
    typed=st.chat_input('예: 87번 학생 몇 점이야? / 그 학생이 틀린 문제는?')
    prompt=typed or chosen
    if prompt:
        if len(prompt)>4000:
            st.warning('질문을 4,000자 이내로 입력해주세요.');return
        history=st.session_state.get('messages',[])
        with st.chat_message('user'):st.markdown(prompt)
        with st.chat_message('assistant'):
            with st.spinner('확인하고 있습니다…'):
                result=answer(prompt,exam,history,st.session_state.get('memory'),settings())
            st.markdown(result['answer'])
            st.caption(f"처리 시간 {result['elapsed_seconds']:.2f}초")
        st.session_state['messages']=history+[{'role':'user','content':prompt},{'role':'assistant','content':result['answer']}]
        st.session_state['memory']=result.get('memory',{})
        st.session_state['last_result']=result

home_page=st.Page(home,title='서비스 소개',icon='📘',default=True)
analysis_page=st.Page(workspace,title='시험 분석',icon='📊',url_path='analysis')
st.navigation([home_page,analysis_page]).run()
