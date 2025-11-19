import streamlit as st
# 제목
st.title("🎉 나의 첫 Streamlit 앱")

# 부제목
st.header("안녕하세요!")

# 텍스트
st.write("Streamlit으로 웹 앱을 만들어봅시다.")

# 사용자 입력
name = st.text_input("이름을 입력하세요")
if name:
    st.success(f"반갑습니다, {name}님! 👋")