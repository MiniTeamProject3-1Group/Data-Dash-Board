import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 페이지 설정
st.set_page_config(page_title="데이터 시각화 앱", page_icon="📊")

# 제목
st.title("📊 간단한 데이터 시각화 대시보드")

# 사이드바
st.sidebar.header("설정")
num_points = st.sidebar.slider("데이터 포인트 수", 10, 100, 50)

# 랜덤 데이터 생성
@st.cache_data
def generate_data(n):
    return pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n),
        'category': np.random.choice(['A', 'B', 'C'], n)
    })

df = generate_data(num_points)

# 데이터 표시
st.subheader("📋 데이터 미리보기")
st.dataframe(df.head())

# 통계
st.subheader("📊 기본 통계")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("평균 X", f"{df['x'].mean():.2f}")
with col2:
    st.metric("평균 Y", f"{df['y'].mean():.2f}")
with col3:
    st.metric("총 데이터", len(df))

# 차트
st.subheader("📈 산점도")
fig = px.scatter(df, x='x', y='y', color='category',
                 title='랜덤 데이터 분포')
st.plotly_chart(fig, use_container_width=True)

# 히스토그램
st.subheader("📊 분포 히스토그램")
chart_type = st.selectbox("변수 선택", ['x', 'y'])
fig2 = px.histogram(df, x=chart_type, nbins=20,
                    title=f'{chart_type} 분포')
st.plotly_chart(fig2, use_container_width=True)
