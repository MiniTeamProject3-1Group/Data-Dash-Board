import streamlit as st
import pandas as pd
import plotly.express as px


st.title("📁 CSV 파일 분석기")

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    # 데이터 읽기
    df = pd.read_csv(uploaded_file)
    st.success("✅ 파일 업로드 성공!")

    # 기본 정보
    st.subheader("📊 데이터 기본 정보")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**행 개수:** {len(df)}")
        st.write(f"**열 개수:** {len(df.columns)}")
    with col2:
        st.write(f"**컬럼:** {', '.join(df.columns)}")

    # 데이터 미리보기
    st.subheader("🔍 데이터 미리보기")
    st.dataframe(df.head(10))

    # 기초 통계
    st.subheader("📈 기초 통계량")
    st.dataframe(df.describe())

    # 컬럼 선택 및 시각화
    st.subheader("📊 시각화")
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("시각화할 컬럼 선택", numeric_cols)
        fig = px.histogram(df, x=selected_col, title=f"{selected_col} 분포")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📎 수치형 컬럼이 없어 히스토그램을 그릴 수 없어요.")
else:
    st.info("👆 CSV 파일을 업로드해주세요!")

    # 샘플 데이터 다운로드
    st.subheader("💾 샘플 데이터 다운로드")
    sample_df = pd.DataFrame(
        {
            "이름": ["철수", "영희", "민수", "지영"],
            "나이": [25, 30, 28, 32],
            "점수": [85, 92, 78, 95],
        }
    )
    csv = sample_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 샘플 CSV 다운로드",
        data=csv,
        file_name="sample_data.csv",
        mime="text/csv",
    )

# 컬럼 나누기
col1, col2, col3 = st.columns(3)

with col1:
    st.write("첫 번째 컬럼")
with col2:
    st.write("두 번째 컬럼")
with col3:
    st.write("세 번째 컬럼")

# 탭 만들기

tab1, tab2 = st.tabs(["데이터", "차트"])
with tab1:
    st.write("데이터 내용")
with tab2:
    st.write("차트 내용")
# 사이드바st.sidebar.title("사이드바")
st.sidebar.write("설정 옵션들")
# Expander (접을 수 있는 영역)with st.expander("자세히 보기"):
st.write("숨겨진 내용")

# 텍스트 입력
text = st.text_input("텍스트 입력")
text_area = st.text_area("긴 텍스트 입력")
# 숫자 입력
number = st.number_input("숫자 입력", min_value=0, max_value=100)
slider = st.slider("슬라이더", 0, 100, 50)
# 선택
option = st.selectbox("선택", ["옵션1", "옵션2", "옵션3"])
multi = st.multiselect("다중 선택", ["A", "B", "C", "D"])
# 날짜
date = st.date_input("날짜 선택")
time = st.time_input("시간 선택")
# 체크박스
check = st.checkbox("동의합니다")
# 버튼
if st.button("클릭하세요"):
    st.write("버튼이 클릭되었습니다!")

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
# 데이터프레임
st.dataframe(df)
# 테이블 (정적)
st.table(df)
# 메트릭
st.metric(label="온도", value="25°C", delta="1.2°C")
# JSON
st.json({'key': 'value'})

# 정보 메시지
st.info("ℹ️ 정보 메시지")
# 성공 메시지
st.success("✅ 성공!")
# 경고 메시지
st.warning("⚠️ 주의!")
# 에러 메시지
st.error("❌ 에러 발생!")
# 풍선
st.balloons()