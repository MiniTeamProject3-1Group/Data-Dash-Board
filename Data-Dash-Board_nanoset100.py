"""
데이터 기반 개인 맞춤 대시보드 (전처리 포함)
패스트캠퍼스 부트캠프 프로젝트

실제 데이터셋 사용 버전
- Kaggle/Hugging Face 데이터 로드
- 자동 전처리 기능
- 결측치 처리, 이상치 제거, 데이터 타입 변환
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="매출 분석 대시보드 (전처리)",
    page_icon="📊",
    layout="wide"
)

# 제목
st.title("📊 데이터 분석 대시보드 (전처리 기능 포함)")
st.markdown("### Kaggle/Hugging Face 데이터셋 활용")
st.markdown("---")

# 사이드바
st.sidebar.header("📁 데이터 소스")
data_source = st.sidebar.radio(
    "데이터 선택:",
    ["샘플 데이터", "파일 업로드", "Kaggle API"]
)

# ==========================================
# 1. 샘플 데이터 생성 함수
# ==========================================
def create_sample_data():
    """샘플 데이터 생성 (기존)"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-05-01', end='2024-10-31', freq='D')
    data = []
    order_id = 1000
    
    products = ['노트북', '스마트폰', '태블릿', '이어폰', '키보드', '마우스', '모니터', '충전기']
    categories = ['전자제품', '전자제품', '전자제품', '액세서리', '액세서리', '액세서리', '전자제품', '액세서리']
    prices = [1200000, 800000, 500000, 150000, 80000, 50000, 350000, 30000]
    regions = ['서울', '경기', '부산', '대구', '인천', '광주', '대전']
    
    for date in dates:
        n_orders = np.random.randint(10, 30)
        for _ in range(n_orders):
            product_idx = np.random.randint(0, len(products))
            quantity = np.random.randint(1, 4)
            
            data.append({
                '주문번호': f'ORD{order_id}',
                '주문일자': date,
                '제품명': products[product_idx],
                '카테고리': categories[product_idx],
                '수량': quantity,
                '단가': prices[product_idx],
                '총금액': prices[product_idx] * quantity,
                '지역': np.random.choice(regions),
                '고객등급': np.random.choice(['일반', 'VIP', '골드'], p=[0.6, 0.3, 0.1])
            })
            order_id += 1
    
    return pd.DataFrame(data)

# ==========================================
# 2. 데이터 전처리 함수 (핵심!)
# ==========================================
def preprocess_data(df, show_steps=True):
    """
    데이터 전처리 함수
    
    처리 내용:
    1. 결측치 처리
    2. 날짜 형식 변환
    3. 숫자 형식 변환
    4. 이상치 제거
    5. 중복 제거
    6. 컬럼명 정리
    """
    
    if show_steps:
        st.markdown("### 🔧 데이터 전처리 진행 중...")
    
    original_rows = len(df)
    
    # Step 1: 컬럼명 정리 (공백, 특수문자 제거)
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    if show_steps:
        st.info("✅ Step 1: 컬럼명 정리 완료")
    
    # Step 2: 결측치 확인 및 처리
    missing_before = df.isnull().sum().sum()
    
    if missing_before > 0:
        if show_steps:
            st.warning(f"⚠️ Step 2: 결측치 {missing_before}개 발견")
            
            # 결측치 처리 옵션
            missing_action = st.selectbox(
                "결측치 처리 방법:",
                ["행 삭제", "평균값으로 채우기", "0으로 채우기", "그대로 두기"]
            )
            
            if missing_action == "행 삭제":
                df = df.dropna()
                st.success(f"✅ {missing_before}개 결측치가 있는 행 삭제 완료")
            elif missing_action == "평균값으로 채우기":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                st.success("✅ 숫자 컬럼의 결측치를 평균값으로 채움")
            elif missing_action == "0으로 채우기":
                df = df.fillna(0)
                st.success("✅ 모든 결측치를 0으로 채움")
    else:
        if show_steps:
            st.success("✅ Step 2: 결측치 없음")
    
    # Step 3: 날짜 컬럼 자동 변환
    date_columns = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', '날짜', '일자', 'day']):
            try:
                df[col] = pd.to_datetime(df[col])
                date_columns.append(col)
            except:
                pass
    
    if show_steps:
        if date_columns:
            st.success(f"✅ Step 3: 날짜 컬럼 변환 완료 ({', '.join(date_columns)})")
        else:
            st.warning("⚠️ Step 3: 날짜 컬럼을 찾지 못했습니다")
    
    # Step 4: 숫자 형식 변환 (문자열로 저장된 숫자)
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # 쉼표 제거 후 숫자로 변환 시도
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('₩', '')
                df[col] = pd.to_numeric(df[col])
            except:
                pass
    
    if show_steps:
        st.success("✅ Step 4: 숫자 형식 변환 완료")
    
    # Step 5: 중복 제거
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        if show_steps:
            st.warning(f"⚠️ Step 5: 중복 {duplicates}개 제거")
    else:
        if show_steps:
            st.success("✅ Step 5: 중복 없음")
    
    # Step 6: 이상치 제거 (IQR 방법)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if show_steps and len(numeric_cols) > 0:
        remove_outliers = st.checkbox("이상치 제거 (IQR 방법)", value=False)
        
        if remove_outliers:
            outliers_removed = 0
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                before = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                outliers_removed += (before - len(df))
            
            st.success(f"✅ Step 6: 이상치 {outliers_removed}개 제거")
    
    # 전처리 요약
    if show_steps:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("원본 데이터", f"{original_rows:,}행")
        with col2:
            st.metric("전처리 후", f"{len(df):,}행")
        with col3:
            removed = original_rows - len(df)
            st.metric("제거된 데이터", f"{removed:,}행", delta=f"{-removed}")
    
    return df

# ==========================================
# 3. 컬럼 매핑 함수 (자동 인식)
# ==========================================
def auto_detect_columns(df):
    """
    데이터프레임의 컬럼을 자동으로 인식
    - 날짜, 금액, 카테고리, 지역 등
    """
    
    mapping = {
        'date': None,
        'amount': None,
        'category': None,
        'region': None,
        'product': None,
        'quantity': None
    }
    
    # 날짜 컬럼 찾기
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            mapping['date'] = col
            break
    
    # 금액 컬럼 찾기
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['price', 'amount', 'total', '금액', '매출', 'sales']):
            if pd.api.types.is_numeric_dtype(df[col]):
                mapping['amount'] = col
                break
    
    # 카테고리 컬럼 찾기
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['category', 'type', '카테고리', '분류']):
            mapping['category'] = col
            break
    
    # 지역 컬럼 찾기
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['region', 'location', 'country', 'city', '지역', '도시']):
            mapping['region'] = col
            break
    
    # 제품 컬럼 찾기
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['product', 'item', 'name', '제품', '상품']):
            mapping['product'] = col
            break
    
    # 수량 컬럼 찾기
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['quantity', 'qty', 'count', '수량', '개수']):
            if pd.api.types.is_numeric_dtype(df[col]):
                mapping['quantity'] = col
                break
    
    return mapping

# ==========================================
# 4. 데이터 로드
# ==========================================
df = None

if data_source == "샘플 데이터":
    df = create_sample_data()
    st.sidebar.success("✅ 샘플 데이터 로드 완료")
    
elif data_source == "파일 업로드":
    uploaded_file = st.sidebar.file_uploader(
        "CSV, Excel, JSON 파일 업로드",
        type=['csv', 'xlsx', 'xls', 'json']
    )
    
    if uploaded_file is not None:
        try:
            # 파일 형식에 따라 로드
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            st.sidebar.success("✅ 파일 업로드 완료")
            
            # 전처리 옵션
            if st.sidebar.checkbox("전처리 실행", value=True):
                df = preprocess_data(df, show_steps=True)
                
        except Exception as e:
            st.sidebar.error(f"❌ 파일 로드 실패: {str(e)}")
            
elif data_source == "Kaggle API":
    st.sidebar.markdown("""
    ### 📥 Kaggle 데이터 다운로드 방법
    
    1. [Kaggle](https://www.kaggle.com) 계정 생성
    2. Account → API → Create New API Token
    3. kaggle.json 파일 다운로드
    4. 아래 데이터셋 선택
    """)
    
    kaggle_dataset = st.sidebar.selectbox(
        "Kaggle 데이터셋:",
        [
            "선택하세요",
            "carrie1/ecommerce-data",
            "aungpyaeap/supermarket-sales",
            "olistbr/brazilian-ecommerce"
        ]
    )
    
    if kaggle_dataset != "선택하세요":
        st.sidebar.info("""
        💡 **수동 다운로드 방법:**
        1. https://www.kaggle.com/datasets/{dataset} 접속
        2. 데이터셋 다운로드
        3. 위의 "파일 업로드" 메뉴에서 업로드
        """.format(dataset=kaggle_dataset))

# ==========================================
# 5. 메인 대시보드
# ==========================================
if df is not None:
    
    # 컬럼 자동 인식
    column_mapping = auto_detect_columns(df)
    
    # 데이터 미리보기
    with st.expander("📋 데이터 미리보기 (원본)", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"총 {len(df):,}개의 레코드, {len(df.columns)}개의 컬럼")
        
        # 데이터 정보
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**컬럼 정보:**")
            st.dataframe(pd.DataFrame({
                '컬럼명': df.columns,
                '데이터 타입': df.dtypes.values,
                '결측치': df.isnull().sum().values
            }), use_container_width=True)
        
        with col2:
            st.markdown("**자동 인식된 컬럼:**")
            for key, value in column_mapping.items():
                if value:
                    st.success(f"✅ {key}: `{value}`")
                else:
                    st.warning(f"⚠️ {key}: 찾지 못함")
    
    # 사이드바 필터
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 필터 옵션")
    
    # 날짜 필터
    date_col = column_mapping['date']
    if date_col:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        date_range = st.sidebar.date_input(
            "날짜 범위",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            mask = (df[date_col] >= pd.to_datetime(date_range[0])) & (df[date_col] <= pd.to_datetime(date_range[1]))
            df_filtered = df[mask].copy()
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
    
    # 카테고리 필터
    category_col = column_mapping['category']
    if category_col and category_col in df_filtered.columns:
        categories = ['전체'] + list(df_filtered[category_col].unique())
        selected_category = st.sidebar.selectbox("카테고리", categories)
        
        if selected_category != '전체':
            df_filtered = df_filtered[df_filtered[category_col] == selected_category]
    
    # KPI 섹션
    st.markdown("### 📈 주요 지표 (KPI)")
    
    amount_col = column_mapping['amount']
    quantity_col = column_mapping['quantity']
    
    if amount_col:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = df_filtered[amount_col].sum()
            st.metric("총 매출", f"₩{total_sales:,.0f}")
        
        with col2:
            avg_sales = df_filtered[amount_col].mean()
            st.metric("평균 주문금액", f"₩{avg_sales:,.0f}")
        
        with col3:
            total_orders = len(df_filtered)
            st.metric("총 주문 수", f"{total_orders:,}")
        
        with col4:
            if quantity_col:
                total_qty = df_filtered[quantity_col].sum()
                st.metric("총 판매량", f"{total_qty:,}")
            else:
                unique_products = df_filtered[column_mapping['product']].nunique() if column_mapping['product'] else 0
                st.metric("제품 종류", f"{unique_products:,}")
    
    st.markdown("---")
    
    # 시각화 섹션
    st.markdown("### 📊 데이터 시각화")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 시계열 분석", "📊 카테고리 분석", "🗺️ 지역 분석", "📉 통계 분석"])
    
    with tab1:
        st.subheader("매출 추이")
        
        if date_col and amount_col:
            # 일별 매출
            daily_sales = df_filtered.groupby(df_filtered[date_col].dt.date)[amount_col].sum().reset_index()
            daily_sales.columns = ['날짜', '매출']
            
            fig = px.line(daily_sales, x='날짜', y='매출', title='일별 매출 추이', markers=True)
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # 월별 매출
            df_filtered['월'] = df_filtered[date_col].dt.to_period('M').astype(str)
            monthly_sales = df_filtered.groupby('월')[amount_col].sum().reset_index()
            
            fig2 = px.bar(monthly_sales, x='월', y=amount_col, title='월별 매출', text=amount_col)
            fig2.update_traces(texttemplate='₩%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("날짜 또는 금액 컬럼을 찾지 못했습니다")
    
    with tab2:
        st.subheader("카테고리/제품 분석")
        
        if category_col and amount_col:
            col1, col2 = st.columns(2)
            
            with col1:
                cat_sales = df_filtered.groupby(category_col)[amount_col].sum().reset_index()
                cat_sales = cat_sales.sort_values(amount_col, ascending=False)
                
                fig = px.pie(cat_sales, values=amount_col, names=category_col, title='카테고리별 매출 비중')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(cat_sales, x=category_col, y=amount_col, title='카테고리별 매출')
                st.plotly_chart(fig, use_container_width=True)
        
        # 제품별 Top 10
        product_col = column_mapping['product']
        if product_col and amount_col:
            st.subheader("Top 10 제품")
            top_products = df_filtered.groupby(product_col)[amount_col].sum().reset_index()
            top_products = top_products.sort_values(amount_col, ascending=False).head(10)
            
            fig = px.bar(top_products, x=product_col, y=amount_col, title='Top 10 제품 매출')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("지역별 분석")
        
        region_col = column_mapping['region']
        
        if region_col and amount_col:
            region_sales = df_filtered.groupby(region_col).agg({
                amount_col: ['sum', 'mean', 'count']
            }).reset_index()
            region_sales.columns = [region_col, '총매출', '평균매출', '주문수']
            region_sales = region_sales.sort_values('총매출', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(region_sales, x=region_col, y='총매출', title='지역별 총 매출')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(region_sales, x='주문수', y='평균매출', size='총매출', 
                               text=region_col, title='지역별 주문수 vs 평균매출')
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("지역 또는 금액 컬럼을 찾지 못했습니다")
    
    with tab4:
        st.subheader("통계 분석")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 기초 통계량")
                stats_df = df_filtered[numeric_cols].describe().T
                st.dataframe(stats_df.round(2), use_container_width=True)
            
            with col2:
                st.markdown("#### 📉 분포도")
                selected_col = st.selectbox("분석할 컬럼", numeric_cols)
                
                fig = px.histogram(df_filtered, x=selected_col, nbins=30, 
                                 title=f'{selected_col} 분포', marginal='box')
                st.plotly_chart(fig, use_container_width=True)
            
            # 상관관계
            if len(numeric_cols) >= 2:
                st.markdown("#### 🔗 상관관계 분석")
                corr_matrix = df_filtered[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                              title='상관관계 히트맵', color_continuous_scale='RdBu_r',
                              zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)
    
    # 데이터 다운로드
    st.markdown("---")
    st.markdown("### 💾 데이터 다운로드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_filtered.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 필터링된 데이터 (CSV)",
            csv,
            "filtered_data.csv",
            "text/csv"
        )
    
    with col2:
        if date_col and amount_col:
            summary = df_filtered.groupby(df_filtered[date_col].dt.date).agg({
                amount_col: ['sum', 'mean', 'count']
            }).reset_index()
            summary_csv = summary.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                "📥 요약 리포트 (CSV)",
                summary_csv,
                "summary_report.csv",
                "text/csv"
            )

else:
    # 데이터가 없을 때
    st.info("👈 좌측 사이드바에서 데이터 소스를 선택해주세요")
    
    st.markdown("""
    ### 🎯 이 대시보드의 기능
    
    #### 📥 데이터 소스
    1. **샘플 데이터**: 즉시 테스트 가능
    2. **파일 업로드**: CSV, Excel, JSON 지원
    3. **Kaggle API**: 실제 데이터셋 연동 (수동 다운로드)
    
    #### 🔧 전처리 기능
    - ✅ 결측치 자동 처리 (삭제/평균/0 채우기)
    - ✅ 날짜 형식 자동 변환
    - ✅ 숫자 형식 자동 변환
    - ✅ 중복 데이터 제거
    - ✅ 이상치 제거 (IQR 방법)
    - ✅ 컬럼명 정리
    
    #### 📊 분석 기능
    - 시계열 분석 (일별/월별)
    - 카테고리별 분석
    - 지역별 분석
    - 통계 분석 (상관관계, 분포)
    
    ### 📚 추천 데이터셋
    
    **Kaggle:**
    - [E-commerce Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
    - [Supermarket Sales](https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales)
    - [Online Retail](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset)
    
    **사용 방법:**
    1. Kaggle 접속 → 데이터셋 다운로드
    2. 좌측 "파일 업로드" 선택
    3. 다운받은 CSV/Excel 파일 업로드
    4. 전처리 실행 → 분석 시작!
    """)

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>📊 데이터 기반 개인 맞춤 대시보드 (전처리 포함) | 패스트캠퍼스 부트캠프 프로젝트</p>
    </div>
    """,
    unsafe_allow_html=True
)


