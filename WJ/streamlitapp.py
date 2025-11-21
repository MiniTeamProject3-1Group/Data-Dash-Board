import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# ----- 한글 폰트 설정 (Windows 기준) -----
mpl.rcParams["font.family"] = "Malgun Gothic"  # 또는 "Malgun Gothic", "NanumGothic" 등
mpl.rcParams["axes.unicode_minus"] = False     # 마이너스 깨짐 방지
# -------------------------------------


# --------------------
# 기본 설정
# --------------------
st.set_page_config(
    page_title="슈퍼마켓 매출 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 공통 색상 팔레트 (BM 인사이트를 위한 일관성)
PRODUCT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c",
                  "#d62728", "#9467bd", "#8c564b"]
GENDER_COLORS = {"Female": "#ff6699", "Male": "#3399ff"}
PAYMENT_COLORS = ["#1f77b4", "#aec7e8", "#d62728"]

# --------------------
# 유틸 함수
# --------------------
@st.cache_data
def load_sample(name: str) -> pd.DataFrame:
    if name == "SuperMarket Analysis":
        path = "data/SuperMarket Analysis.csv"
    else:
        path = "data/supermarket_sales.csv"

    df = pd.read_csv(path)

    if "Sales" in df.columns and "Total" not in df.columns:
        df = df.rename(columns={"Sales": "Total"})

    df = preprocess_supermarket(df)
    return df

def preprocess_supermarket(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 컬럼 이름이 조금씩 다른 다양한 판매 CSV를
      우리가 쓰는 공통 스키마에 최대한 맞춰줌.
    - 그 후 Date/Time/avg_ticket 등을 계산.
    """
    df = df.copy()
    cols = set(df.columns)

    # --- 매출 금액: Total ---
    if "Total" not in cols:
        for cand in ["Sales", "Sale", "Amount", "Revenue",
                     "RETAIL SALES", "Retail Sales"]:
            if cand in cols:
                df["Total"] = df[cand]
                break

    # --- 수량: Quantity ---
    if "Quantity" not in cols:
        for cand in ["Qty", "QTY", "quantity"]:
            if cand in cols:
                df["Quantity"] = df[cand]
                break

    # --- 주문/거래 ID: Invoice ID ---
    if "Invoice ID" not in cols:
        for cand in ["InvoiceID", "Invoice_Id", "Order ID",
                     "OrderID", "BillNo", "Bill No",
                     "Customer ID", "Cust ID"]:
            if cand in cols:
                df["Invoice ID"] = df[cand]
                break
        # 위 후보들 다 없으면 인덱스로라도 생성 (Retail & warehouse 파일용)
        if "Invoice ID" not in df.columns:
            df["Invoice ID"] = np.arange(len(df))

    # --- 날짜: Date ---
    if "Date" not in cols:
        date_col = None
        for cand in ["Order Date", "Order_Date", "InvoiceDate",
                     "Invoice Date", "date", "Date"]:
            if cand in df.columns:
                date_col = cand
                break

        if date_col is not None:
            df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        # YEAR + MONTH 조합으로 월 단위 Date 생성 (Retail & warehouse 파일용)
        elif "YEAR" in df.columns and "MONTH" in df.columns:
            df["Date"] = pd.to_datetime(
                df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01",
                errors="coerce"
            )

    # --- 상품 라인/카테고리: Product line ---
    cols = set(df.columns)  # 위에서 컬럼이 늘어났으니 한 번 갱신
    if "Product line" not in cols:
        for cand in ["Product line", "Category", "Sub-Category",
                     "Product Name", "Product", "Item Description", "ITEM DESCRIPTION",
                     "Item Type", "ITEM TYPE"]:
            if cand in df.columns:
                df["Product line"] = df[cand]
                break

    # --- 고객 유형/세그먼트: Customer type ---
    if "Customer type" not in df.columns:
        for cand in ["Customer type", "Segment", "Customer Segment", "CustType"]:
            if cand in df.columns:
                df["Customer type"] = df[cand]
                break

    # --- 결제/배송 모드: Payment ---
    if "Payment" not in df.columns:
        for cand in ["Payment", "Payment Method", "PaymentMode",
                     "Pay Mode", "Ship Mode"]:
            if cand in df.columns:
                df["Payment"] = df[cand]
                break

    # --- 지역/도시: City ---
    if "City" not in df.columns:
        for cand in ["City", "Region", "State"]:
            if cand in df.columns:
                df["City"] = df[cand]
                break

    # --- 지점/창고: Branch ---
    if "Branch" not in df.columns:
        for cand in ["Branch", "Store", "Warehouse", "State", "Region"]:
            if cand in df.columns:
                df["Branch"] = df[cand]
                break

    # --- 이익: gross income ---
    if "gross income" not in df.columns:
        for cand in ["Profit", "Gross Income", "gross_income"]:
            if cand in df.columns:
                df["gross income"] = df[cand]
                break

    # --- 평점: Rating (있으면 매핑) ---
    if "Rating" not in df.columns:
        for cand in ["rating", "Rating", "Score", "Customer Rating"]:
            if cand in df.columns:
                df["Rating"] = df[cand]
                break

    # -------- Date/Time/avg_ticket 계산 --------
    # Date → year_month, day_name
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["year_month"] = df["Date"].dt.to_period("M").astype(str)
        df["day_name"] = df["Date"].dt.day_name()

    # Time → hour / period
    if "Time" in df.columns:
        t = pd.to_datetime(df["Time"], format="%I:%M:%S %p", errors="coerce")
        mask = t.isna()
        if mask.any():
            t.loc[mask] = pd.to_datetime(
                df.loc[mask, "Time"], format="%H:%M", errors="coerce"
            )
        df["hour"] = t.dt.hour

        def hour_to_period(h):
            if pd.isna(h):
                return "Unknown"
            h = int(h)
            if 6 <= h < 11:
                return "Morning"
            elif 11 <= h < 14:
                return "Lunch"
            elif 14 <= h < 18:
                return "Afternoon"
            elif 18 <= h < 22:
                return "Evening"
            else:
                return "Night"

        df["period"] = df["hour"].apply(hour_to_period)

    # 객단가
    if "Total" in df.columns and "Quantity" in df.columns:
        df["avg_ticket"] = df["Total"] / df["Quantity"]

    return df

def is_supermarket_schema(df: pd.DataFrame) -> bool:
    """
    대시보드가 돌아갈 수 있는 최소 조건만 체크.
    - Total : 매출 금액
    - Invoice ID : 거래 단위 식별자 (없으면 preprocess에서 만들어 줌)
    - Date 또는 year_month : 시간 분석용
    """
    cols = set(df.columns)

    if "Total" not in cols:
        return False
    if "Invoice ID" not in cols:
        return False

    has_date = "Date" in cols
    has_year_month = "year_month" in cols

    return has_date or has_year_month

def generate_bm_insights(df: pd.DataFrame) -> str:
    """
    Overview 탭 KPI + 도시별/지점별 매출 구조에 맞춘 BM 인사이트 생성
    - Total, Invoice ID, City, Branch, Rating, avg_ticket 컬럼을 우선적으로 사용
    """
    if df is None or df.empty:
        return "현재 필터 조건에서는 데이터가 없습니다. 필터를 조정한 뒤 다시 확인해 보세요."

    insights = []
    cols = set(df.columns)

    # 0) 기본 KPI
    total_sales = df["Total"].sum() if "Total" in cols else None
    n_orders = df["Invoice ID"].nunique() if "Invoice ID" in cols else None
    avg_rating = df["Rating"].mean() if "Rating" in cols else None
    avg_ticket = df["avg_ticket"].mean() if "avg_ticket" in cols else None

    if total_sales is not None and n_orders is not None:
        insights.append(
            f"- 현재 필터 기준 **총 매출은 약 {total_sales:,.0f}원**, **거래 수는 {n_orders:,}건**입니다.  \n"
            "  → 이 구간을 기준으로 목표 매출/주문 수를 설정하고, 프로모션 효과를 비교할 수 있습니다."
        )

    if avg_ticket is not None:
        insights.append(
            f"- 주문 1건당 평균 객단가(평균 매출)는 **약 {avg_ticket:,.0f}원** 수준입니다.  \n"
            "  → 세트 상품 구성, 업셀링(고가 옵션 제안) 등으로 객단가를 올릴 수 있는 여지를 검토해 볼 수 있습니다."
        )

    if avg_rating is not None:
        insights.append(
            f"- 전체 평균 평점은 **{avg_rating:.2f}점**입니다.  \n"
            "  → 평점이 높게 유지되는 구간의 상품/서비스 특징을 분석해 다른 지점·카테고리에도 확산할 수 있습니다."
        )

    # 1) 도시별 매출 인사이트 (City별 총 매출 그래프용)
    if "City" in cols and "Total" in cols:
        city_sales = df.groupby("City")["Total"].sum().sort_values(ascending=False)
        if len(city_sales) > 0:
            top_city = city_sales.index[0]
            top_city_val = city_sales.iloc[0]
            top_ratio = top_city_val / city_sales.sum() * 100 if city_sales.sum() > 0 else 0

            txt = (
                f"- **도시별 매출**을 보면 **{top_city}**가 가장 높으며, "
                f"총 매출은 약 **{top_city_val:,.0f}원**으로 전체의 **{top_ratio:.1f}%**를 차지합니다.  \n"
                "  → 이 도시를 핵심 거점으로 보고 재고·마케팅·인력을 우선 배치하는 전략을 고려할 수 있습니다."
            )
            insights.append(txt)

            # 상위/하위 도시 간 격차
            if len(city_sales) >= 2:
                bottom_city = city_sales.index[-1]
                bottom_val = city_sales.iloc[-1]
                if top_city_val >= bottom_val * 1.5 and bottom_val > 0:
                    insights.append(
                        f"- 상위 도시(**{top_city}**)와 하위 도시(**{bottom_city}**)의 매출 격차가 큽니다.  \n"
                        "  → 하위 도시는 프로모션, 진열 개선, 가격 정책 등을 집중적으로 테스트해 볼 후보입니다."
                    )

    # 2) 지점별 매출 인사이트 (Branch별 총 매출 그래프용)
    if "Branch" in cols and "Total" in cols:
        branch_sales = df.groupby("Branch")["Total"].sum().sort_values(ascending=False)
        if len(branch_sales) > 0:
            top_branch = branch_sales.index[0]
            top_branch_val = branch_sales.iloc[0]
            br_ratio = top_branch_val / branch_sales.sum() * 100 if branch_sales.sum() > 0 else 0

            insights.append(
                f"- **지점별 매출**에서는 **{top_branch} 지점**이 가장 높은 매출 "
                f"(**약 {top_branch_val:,.0f}원**, 비중 **{br_ratio:.1f}%**)을 기록하고 있습니다.  \n"
                "  → 이 지점의 운영 방식(상품 믹스, 직원 구성, 프로모션)을 벤치마킹해 다른 지점에 확산할 수 있습니다."
            )

    # 3) 도시·지점별 평점 인사이트 (Rating이 있을 때만)
    if "Rating" in cols and "City" in cols:
        city_rating = df.groupby("City")["Rating"].mean().sort_values(ascending=False)
        if len(city_rating) > 0:
            best_city = city_rating.index[0]
            best_city_rating = city_rating.iloc[0]
            insights.append(
                f"- **도시별 평점** 기준으로는 **{best_city}**의 평균 평점이 **{best_city_rating:.2f}점**으로 가장 높습니다.  \n"
                "  → 이 도시에서 잘 팔리는 상품/서비스를 기준으로, 다른 지역의 상품 구성과 CS 전략을 조정할 수 있습니다."
            )

    if not insights:
        return "현재 필터 조건에서는 뚜렷한 패턴이 잘 보이지 않습니다. 다른 필터 조합으로 다시 확인해 보세요."

    return "\n".join(insights)


# 🔹 룰 기반 BM 인사이트 생성 함수 (df_filtered 기준으로 매번 자동 생성)
def generate_bm_insights2(df: pd.DataFrame) -> str:
    """현재 필터가 적용된 df를 기반으로 BM 아이디어를 자동 생성"""
    if df is None or df.empty:
        return "현재 필터 조건에서는 데이터가 없습니다. 필터를 조정한 뒤 다시 확인해 보세요."

    insights = []

    # 1) 성별 매출 비중
    if {"Gender", "Total"}.issubset(df.columns):
        gender_sales = df.groupby("Gender")["Total"].sum().sort_values(ascending=False)
        if not gender_sales.empty and gender_sales.sum() > 0:
            top_gender = gender_sales.index[0]
            ratio = gender_sales.iloc[0] / gender_sales.sum() * 100
            insights.append(
                f"- 현재 필터 기준 매출의 약 **{ratio:.1f}%**가 **{top_gender} 고객**에서 발생합니다.  \n"
                f"  → 이 타깃을 중심으로 한 프로모션/추천 상품 구성이 효과적일 수 있습니다."
            )

    # 2) 상품 라인 TOP 1
    if {"Product line", "Total"}.issubset(df.columns):
        pl_sales = df.groupby("Product line")["Total"].sum().sort_values(ascending=False)
        if not pl_sales.empty:
            top_pl = pl_sales.index[0]
            top_pl_val = pl_sales.iloc[0]
            insights.append(
                f"- 매출 1위 상품 라인은 **{top_pl}** (총 매출 약 **{top_pl_val:,.0f}원**)입니다.  \n"
                "  → 이 카테고리를 메인 배너/추천 섹션에 노출하고, 관련 상품을 묶음 판매하는 BM을 고민해볼 수 있습니다."
            )

    # 3) 시간대별 매출 피크
    if {"period", "Total"}.issubset(df.columns) and not df["period"].isna().all():
        per_sales = df.groupby("period")["Total"].sum().sort_values(ascending=False)
        if not per_sales.empty:
            peak_period = per_sales.index[0]
            insights.append(
                f"- 가장 매출이 높은 시간대는 **{peak_period}**입니다.  \n"
                "  → 해당 시간대에 맞춰 쿠폰/푸시 알림/라이브커머스 등을 집중 배치하는 전략이 유효합니다."
            )

    # 4) 요일별 매출 편차
    if {"day_name", "Total"}.issubset(df.columns):
        dow = df.groupby("day_name")["Total"].sum()
        if len(dow) >= 2 and dow.max() > 0:
            best_day = dow.idxmax()
            worst_day = dow.idxmin()
            # 편차가 어느 정도 이상일 때만 코멘트
            if dow.max() >= dow.min() * 1.3:
                insights.append(
                    f"- 요일별 매출 차이가 큽니다. **{best_day}**가 가장 강하고, **{worst_day}**가 가장 약합니다.  \n"
                    "  → 약한 요일에는 한정 프로모션을 걸고, 강한 요일에는 재고/인력을 집중 배치하는 BM을 설계할 수 있습니다."
                )

    # 5) 멤버십 고객 비중
    if {"Customer type", "Total"}.issubset(df.columns):
        ct = df["Customer type"].value_counts(normalize=True) * 100
        if not ct.empty:
            top_ct = ct.index[0]
            top_ct_ratio = ct.iloc[0]
            insights.append(
                f"- 고객 유형 중 **{top_ct}**가 {top_ct_ratio:.1f}%로 가장 큰 비중을 차지합니다.  \n"
                "  → 이 고객군을 위한 전용 혜택(멤버십 등급, 장바구니 쿠폰, 적립 강화)을 강화하면 재방문과 객단가 상승에 도움이 될 수 있습니다."
            )

    if not insights:
        return "현재 필터 조건에서는 뚜렷한 패턴이 잘 보이지 않습니다. 다른 필터 조합으로 다시 확인해 보세요."
    return "\n".join(insights)


# --------------------
# 상단 헤더
# --------------------
st.title("미니프로젝트3_1조")
st.header("슈퍼마켓 - 💸 매출 분석 대시보드")
st.write("슈퍼마켓 데이터를 기반으로, **월별 매출·고객·상품·시간대 인사이트를 통해 BM 전략**을 세울 수 있도록 만든 대시보드입니다.")

# --------------------
# 사이드바: 데이터 소스 & 필터
# --------------------
st.sidebar.title("데이터 소스")

data_source = st.sidebar.radio(
    "데이터 선택",
    ("샘플: SuperMarket Analysis", "샘플: supermarket_sales", "CSV 업로드")
)

uploaded_file = None
df = None
supermarket_mode = False

if data_source.startswith("샘플"):
    sample_name = "SuperMarket Analysis" if "Analysis" in data_source else "supermarket_sales"
    df = load_sample(sample_name)
    supermarket_mode = True
else:
    uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded_file is not None:
        # 1) 원본 읽기
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ 파일 업로드 성공!")

        # 2) 항상 먼저 공통 전처리 적용
        df = preprocess_supermarket(raw_df)

        # 3) 전처리된 df 기준으로 최소 스키마 체크
        supermarket_mode = is_supermarket_schema(df)

        # 4) 매출 컬럼조차 없으면 경고
        if not supermarket_mode:
            st.warning(
                "이 CSV에서는 매출 금액( Total )이나 날짜 정보를 찾지 못해서 "
                "기본 정보만 보여주고 있어. 컬럼 이름을 한 번 확인해줘."
            )
    else:
        st.sidebar.info("CSV 파일을 업로드하면 분석이 시작됩니다.")


# --------------------
# 본문
# --------------------
if df is not None:

    # 기본 정보
    with st.expander("📊 데이터 기본 정보", expanded=False):
        c1, c2 = st.columns([1, 3])
        with c1:
            st.write(f"**행 개수:** {len(df):,}")
            st.write(f"**열 개수:** {len(df.columns)}")
        with c2:
            st.write("**컬럼 목록:**")
            st.write(", ".join(df.columns))

    if supermarket_mode:

        # ---- 필터 UI (시각적으로 정리) ----
        st.sidebar.markdown("---")
        st.sidebar.subheader("필터")

        # 날짜 필터
        if "Date" in df.columns:
            min_date = df["Date"].min()
            max_date = df["Date"].max()
            date_range = st.sidebar.date_input(
                "날짜 범위",
                [min_date, max_date]
            )
        else:
            date_range = None

        with st.sidebar.expander("지역 / 지점", expanded=True):
            def cat_filter(col_name: str):
                options = sorted(df[col_name].dropna().unique())
                return st.multiselect(col_name, options, default=options)

            city_selected = cat_filter("City") if "City" in df.columns else None
            branch_selected = cat_filter("Branch") if "Branch" in df.columns else None

        with st.sidebar.expander("고객 / 상품 / 결제", expanded=False):
            def cat_filter_opt(col_name: str):
                if col_name not in df.columns:
                    return None
                options = sorted(df[col_name].dropna().unique())
                return st.multiselect(col_name, options, default=options)

            ctype_selected = cat_filter_opt("Customer type")
            gender_selected = cat_filter_opt("Gender")
            pline_selected = cat_filter_opt("Product line")
            pay_selected = cat_filter_opt("Payment")

        # ---- 필터 적용 ----
        df_filtered = df.copy()

        if date_range is not None and len(date_range) == 2:
            start, end = date_range
            df_filtered = df_filtered[(df_filtered["Date"] >= pd.to_datetime(start))
                                      & (df_filtered["Date"] <= pd.to_datetime(end))]

        def apply_cat_filter(df_, col, selected):
            if col in df_.columns and selected is not None:
                df_ = df_[df_[col].isin(selected)]
            return df_

        df_filtered = apply_cat_filter(df_filtered, "City", city_selected)
        df_filtered = apply_cat_filter(df_filtered, "Branch", branch_selected)
        df_filtered = apply_cat_filter(df_filtered, "Customer type", ctype_selected)
        df_filtered = apply_cat_filter(df_filtered, "Gender", gender_selected)
        df_filtered = apply_cat_filter(df_filtered, "Product line", pline_selected)
        df_filtered = apply_cat_filter(df_filtered, "Payment", pay_selected)

        st.caption(f"필터 적용 후 행 개수: {len(df_filtered):,} 행")

        # --------------------
        # 탭 구성
        # --------------------
        tab_overview, tab_stats, tab_viz, tab_corr = st.tabs(
            ["Overview", "통계 분석", "시각화", "상관관계"]
        )

        # ===== Overview =====
        with tab_overview:
            st.subheader("📌 주요 지표 (KPI)")

            total_sales = df_filtered["Total"].sum()
            avg_sales = df_filtered["Total"].mean()
            n_orders = df_filtered["Invoice ID"].nunique()
            avg_rating = df_filtered["Rating"].mean()

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("총 매출", f"₩{total_sales:,.0f}")
            with k2:
                st.metric("평균 매출(주문당)", f"₩{avg_sales:,.0f}")
            with k3:
                st.metric("거래 수", f"{n_orders:,} 건")
            with k4:
                st.metric("평균 평점", f"{avg_rating:.2f}")

            st.markdown("")

            # KPI 바로 아래에 지역/지점 매출 배치 (한눈에 BM 인사이트용)
            c1, c2 = st.columns(2)

            with c1:
                if "City" in df_filtered.columns:
                    st.markdown("### 🏙️ 도시별 매출")
                    city_sales = df_filtered.groupby("City")["Total"].sum().reset_index()
                    fig = px.bar(
                        city_sales,
                        x="City",
                        y="Total",
                        text_auto=".2s",
                        title="City별 총 매출",
                        color="City",
                        color_discrete_sequence=PRODUCT_COLORS
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                if "Branch" in df_filtered.columns:
                    st.markdown("### 🏬 지점별 매출")
                    b_sales = df_filtered.groupby("Branch")["Total"].sum().reset_index()
                    fig2 = px.bar(
                        b_sales,
                        x="Branch",
                        y="Total",
                        text_auto=".2s",
                        title="Branch별 총 매출",
                        color="Branch",
                        color_discrete_sequence=PRODUCT_COLORS
                    )
                    fig2.update_layout(showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)

            # 🔥 여기서 BM 자동 코멘트 출력
            st.markdown("---")
            st.markdown("### 💡 이 데이터에서 생각해볼 수 있는 BM 아이디어")
            bm_text = generate_bm_insights(df_filtered)
            st.markdown(bm_text)

            st.markdown("### 🔍 데이터 미리보기")
            st.dataframe(df_filtered.head(20), use_container_width=True)

        # ===== 통계 분석 =====
        with tab_stats:
            st.subheader("📈 매출/고객 요약 통계")

            c1, c2, c3 = st.columns(3)

            # 매출 통계
            with c1:
                st.markdown("#### 💰 매출 통계")
                sales_stats = pd.DataFrame({
                    "지표": ["총매출", "평균 매출(주문당)", "매출 중앙값"],
                    "값": [
                        f"{df_filtered['Total'].sum():,.0f}",
                        f"{df_filtered['Total'].mean():,.0f}",
                        f"{df_filtered['Total'].median():,.0f}",
                    ]
                })
                st.table(sales_stats)

            # 이익 통계
            with c2:
                if "gross income" in df_filtered.columns:
                    st.markdown("#### 📊 이익 통계")
                    profit_stats = pd.DataFrame({
                        "지표": ["총 이익", "평균 이익(주문당)"],
                        "값": [
                            f"{df_filtered['gross income'].sum():,.0f}",
                            f"{df_filtered['gross income'].mean():,.0f}",
                        ]
                    })
                    st.table(profit_stats)

            # 고객 구조
            with c3:
                st.markdown("#### 🙋‍♀️ 고객 구조")
                info_rows = []
                if "Customer type" in df_filtered.columns:
                    ct = df_filtered["Customer type"].value_counts(normalize=True) * 100
                    for k, v in ct.items():
                        info_rows.append([f"Customer type: {k}", f"{v:.1f}%"])
                if "Gender" in df_filtered.columns:
                    gd = df_filtered["Gender"].value_counts(normalize=True) * 100
                    for k, v in gd.items():
                        info_rows.append([f"Gender: {k}", f"{v:.1f}%"])
                info_rows.append(["평균 평점", f"{df_filtered['Rating'].mean():.2f}"])
                st.table(pd.DataFrame(info_rows, columns=["항목", "값"]))

            st.markdown("---")
            st.markdown("### 📆 월별 매출 요약 (BM 설계용)")

            if "year_month" in df_filtered.columns:
                monthly = (df_filtered
                           .groupby("year_month")
                           .agg(
                               total_sales=("Total", "sum"),
                               avg_ticket=("avg_ticket", "mean"),
                               n_orders=("Invoice ID", "nunique")
                           )
                           .reset_index()
                           .sort_values("year_month"))

                monthly["mom_growth"] = monthly["total_sales"].pct_change() * 100

                best_idx = monthly["total_sales"].idxmax()
                worst_idx = monthly["total_sales"].idxmin()
                best_month = monthly.loc[best_idx, "year_month"]
                best_value = monthly.loc[best_idx, "total_sales"]
                worst_month = monthly.loc[worst_idx, "year_month"]
                worst_value = monthly.loc[worst_idx, "total_sales"]

                mk1, mk2, mk3 = st.columns(3)
                with mk1:
                    st.metric("월별 평균 매출", f"₩{monthly['total_sales'].mean():,.0f}")
                with mk2:
                    st.metric("최고 매출 월", best_month, f"₩{best_value:,.0f}")
                with mk3:
                    st.metric("최저 매출 월", worst_month, f"₩{worst_value:,.0f}")

                show_df = monthly.copy()
                show_df["total_sales"] = show_df["total_sales"].map(lambda x: f"{x:,.0f}")
                show_df["avg_ticket"] = show_df["avg_ticket"].map(lambda x: f"{x:,.0f}")
                show_df["mom_growth"] = show_df["mom_growth"].map(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
                )
                show_df.rename(columns={
                    "year_month": "월",
                    "total_sales": "총매출",
                    "avg_ticket": "평균 객단가",
                    "n_orders": "주문 수",
                    "mom_growth": "전월 대비 성장률"
                }, inplace=True)

                st.table(show_df)

        # ===== 시각화 =====
        with tab_viz:
            st.subheader("📊 시각화 대시보드")

            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["매출 구조", "분포 분석", "시간 분석"])

            # --- 매출 구조 ---
            with viz_tab1:
                c1, c2 = st.columns(2)

                if "Product line" in df_filtered.columns:
                    # 상품 라인별 매출
                    with c1:
                        st.markdown("#### 상품 라인별 매출 (Bar)")
                        pl_sales = df_filtered.groupby("Product line")["Total"].sum().reset_index()
                        fig = px.bar(
                            pl_sales,
                            x="Product line",
                            y="Total",
                            text_auto=".2s",
                            title="Product line별 총 매출",
                            color="Product line",
                            color_discrete_sequence=PRODUCT_COLORS
                        )
                        fig.update_layout(showlegend=False, xaxis_tickangle=-25)
                        st.plotly_chart(fig, use_container_width=True)

                    # 상품 라인 매출 비중
                    with c2:
                        st.markdown("#### 상품 라인 매출 비중 (Pie)")
                        fig_p = px.pie(
                            pl_sales,
                            names="Product line",
                            values="Total",
                            title="Product line 매출 비중",
                            color="Product line",
                            color_discrete_sequence=PRODUCT_COLORS
                        )
                        st.plotly_chart(fig_p, use_container_width=True)

                st.markdown("---")
                c3, c4 = st.columns(2)

                # 결제 수단 비율
                if "Payment" in df_filtered.columns:
                    with c3:
                        st.markdown("#### 결제 수단 비율 (Pie)")
                        pay_cnt = df_filtered["Payment"].value_counts().reset_index()
                        pay_cnt.columns = ["Payment", "count"]
                        fig_pay = px.pie(
                            pay_cnt,
                            names="Payment",
                            values="count",
                            title="결제 수단 비율",
                            color="Payment",
                            color_discrete_sequence=PAYMENT_COLORS
                        )
                        st.plotly_chart(fig_pay, use_container_width=True)

                # 고객 유형 × 성별
                if "Customer type" in df_filtered.columns and "Gender" in df_filtered.columns:
                    with c4:
                        st.markdown("#### 고객 유형 × 성별 (Bar)")
                        ct_gender = (df_filtered
                                     .groupby(["Customer type", "Gender"])
                                     ["Invoice ID"].nunique()
                                     .reset_index())
                        ct_gender.rename(columns={"Invoice ID": "orders"}, inplace=True)
                        fig_cg = px.bar(
                            ct_gender,
                            x="Customer type",
                            y="orders",
                            color="Gender",
                            barmode="group",
                            title="Customer type × Gender별 거래 수",
                            color_discrete_map=GENDER_COLORS
                        )
                        st.plotly_chart(fig_cg, use_container_width=True)

                        
            # 🔥 여기서 BM 자동 코멘트 출력
            st.markdown("---")
            st.markdown("### 💡 이 데이터에서 생각해볼 수 있는 BM 아이디어")
            bm_text = generate_bm_insights2(df_filtered)
            st.markdown(bm_text)

            # --- 분포 분석 ---
            with viz_tab2:
                st.markdown("#### 가격 / 평점 / 객단가 분포")

                c1, c2 = st.columns(2)

                if "Unit price" in df_filtered.columns:
                    with c1:
                        st.markdown("##### Unit price 분포 (Histogram)")
                        fig_up = px.histogram(
                            df_filtered,
                            x="Unit price",
                            nbins=30,
                            title="Unit price 분포"
                        )
                        st.plotly_chart(fig_up, use_container_width=True)

                if "Rating" in df_filtered.columns:
                    with c2:
                        st.markdown("##### Rating 분포 (Histogram)")
                        fig_rt = px.histogram(
                            df_filtered,
                            x="Rating",
                            nbins=20,
                            title="Rating 분포"
                        )
                        st.plotly_chart(fig_rt, use_container_width=True)

                st.markdown("---")

                if "avg_ticket" in df_filtered.columns and "Product line" in df_filtered.columns:
                    st.markdown("##### 상품 라인별 객단가 분포 (Boxplot)")
                    fig_box = px.box(
                        df_filtered,
                        x="Product line",
                        y="avg_ticket",
                        points="all",
                        title="Product line별 avg_ticket 분포",
                        color="Product line",
                        color_discrete_sequence=PRODUCT_COLORS
                    )
                    fig_box.update_layout(xaxis_tickangle=-25)
                    st.plotly_chart(fig_box, use_container_width=True)

            # --- 시간 분석 ---
            with viz_tab3:
                st.markdown("#### 시간 기반 매출 분석")

                # 🔁 분석 단위: 월별 / 요일별 / 시간대
                view_type = st.radio(
                    "분석 단위 선택",
                    ["월별", "요일별", "시간대"],
                    horizontal=True
                )

                # 🔁 교집합으로 보고 싶은 기준 (여러 개 선택 가능)
                seg_candidates = ["Gender", "Customer type", "City", "Branch", "Product line"]
                seg_dims = st.multiselect(
                    "교집합으로 보고 싶은 기준 선택 (여러 개 선택 가능)",
                    seg_candidates,
                    default=["Gender", "Customer type"]  # 기본: 성별 + 고객유형
                )

                # 🔁 기준별 세부 값 선택
                seg_values = {}
                for dim in seg_dims:
                    if dim in df_filtered.columns:
                        options = sorted(df_filtered[dim].dropna().unique())
                        chosen = st.multiselect(
                            f"{dim} 값 선택",
                            options,
                            default=options,
                            key=f"segval_{dim}"
                        )
                        seg_values[dim] = chosen

                # 공통: Date 기반 전처리
                df_time = None
                if "Date" in df_filtered.columns:
                    df_time = df_filtered.copy()
                    df_time["Date"] = pd.to_datetime(df_time["Date"], errors="coerce")
                    df_time = df_time.dropna(subset=["Date"])
                    df_time["year_month"] = df_time["Date"].dt.to_period("M").astype(str)
                    if "day_name" not in df_time.columns:
                        df_time["day_name"] = df_time["Date"].dt.day_name()

                def apply_seg_filters(df_: pd.DataFrame) -> pd.DataFrame:
                    df_res = df_.copy()
                    for dim, vals in seg_values.items():
                        if dim in df_res.columns and vals:
                            df_res = df_res[df_res[dim].isin(vals)]
                    return df_res

                if df_time is not None:
                    df_time = apply_seg_filters(df_time)
                base_df_for_time = apply_seg_filters(df_filtered)

                def add_segment_label(df_seg: pd.DataFrame, seg_cols: list) -> pd.DataFrame:
                    if not seg_cols:
                        df_seg["segment"] = "전체"
                    else:
                        def _row_to_segment(row):
                            parts = []
                            for col in seg_cols:
                                if col in row and pd.notna(row[col]):
                                    parts.append(f"{col}: {row[col]}")
                            return " / ".join(parts) if parts else "전체"
                        df_seg["segment"] = df_seg.apply(_row_to_segment, axis=1)
                    return df_seg

                # ========= 1) 월별 분석 =========
                if view_type == "월별":
                    if df_time is None or df_time.empty:
                        st.info("월별 분석을 위한 데이터가 없습니다. (날짜/필터/세그먼트 선택을 확인해 주세요)")
                    else:
                        group_cols = ["year_month"]
                        for col in seg_dims:
                            if col in df_time.columns:
                                group_cols.append(col)

                        monthly = (
                            df_time
                            .groupby(group_cols)["Total"]
                            .sum()
                            .reset_index()
                        )
                        monthly = add_segment_label(monthly, seg_dims)

                        order = sorted(monthly["year_month"].unique())
                        monthly["year_month"] = pd.Categorical(
                            monthly["year_month"], categories=order, ordered=True
                        )

                        fig_m = px.line(
                            monthly,
                            x="year_month",
                            y="Total",
                            color="segment",
                            markers=True,
                            title="월별 총 매출 (선택한 교집합 기준별)",
                        )
                        fig_m.update_layout(
                            xaxis_title="월",
                            yaxis_title="총 매출",
                        )
                        st.plotly_chart(fig_m, use_container_width=True)
                                            
                    # ========= 2) 요일별 분석 =========
                elif view_type == "요일별":
                        if df_time is None or "day_name" not in df_time.columns or df_time.empty:
                            st.info("요일별 분석을 위한 데이터가 없습니다. (날짜/필터/세그먼트 선택을 확인해 주세요)")
                        else:
                            group_cols = ["day_name"]
                            for col in seg_dims:
                                if col in df_time.columns:
                                    group_cols.append(col)

                            dow = (
                                df_time
                                .groupby(group_cols)["Total"]
                                .sum()
                                .reset_index()
                            )
                            dow = add_segment_label(dow, seg_dims)

                            # ✅ 요일 고정 순서 정의
                            day_order = ["Monday", "Tuesday", "Wednesday",
                                        "Thursday", "Friday", "Saturday", "Sunday"]

                            # 카테고리형 + 정렬
                            dow["day_name"] = pd.Categorical(
                                dow["day_name"], categories=day_order, ordered=True
                            )
                            dow = dow.sort_values("day_name")

                            # ✅ Plotly에 요일 순서 직접 전달
                            fig_dow = px.line(
                                dow,
                                x="day_name",
                                y="Total",
                                color="segment",
                                markers=True,
                                title="요일별 총 매출 (선택한 교집합 기준별)",
                                category_orders={"day_name": day_order},  # ← 이 줄이 핵심
                            )
                            fig_dow.update_layout(
                                xaxis_title="요일",
                                yaxis_title="총 매출",
                            )
                            st.plotly_chart(fig_dow, use_container_width=True)


                # ========= 3) 시간대 분석 =========
                else:  # view_type == "시간대"
                    if "period" not in base_df_for_time.columns or base_df_for_time.empty:
                        st.info("시간대 분석을 위한 데이터가 없습니다. (Time/필터/세그먼트 선택을 확인해 주세요)")
                    else:
                        group_cols = ["period"]
                        for col in seg_dims:
                            if col in base_df_for_time.columns:
                                group_cols.append(col)

                        ht = (
                            base_df_for_time
                            .groupby(group_cols)["Total"]
                            .sum()
                            .reset_index()
                        )
                        ht = add_segment_label(ht, seg_dims)

                        period_order = ["Morning", "Lunch", "Afternoon",
                                        "Evening", "Night", "Unknown"]
                        ht["period"] = pd.Categorical(
                            ht["period"], categories=period_order, ordered=True
                        )
                        ht = ht.sort_values("period")

                        fig_t = px.line(
                            ht,
                            x="period",
                            y="Total",
                            color="segment",
                            markers=True,
                            title="시간대별 총 매출 (선택한 교집합 기준별)",
                        )
                        fig_t.update_layout(
                            xaxis_title="시간대",
                            yaxis_title="총 매출",
                        )
                        st.plotly_chart(fig_t, use_container_width=True)

        # ===== 상관관계 =====
        with tab_corr:
            st.subheader("📉 수치형 변수 상관관계 분석")

            preferred_cols = [
                "Unit price",       # 개당 가격
                "Quantity",         # 수량
                "Total",            # 총 매출
                "gross income",     # 이익
                "Rating",           # 평점
                "avg_ticket",       # 객단가
            ]

            numeric_all = df_filtered.select_dtypes(include=["float64", "int64"]).columns.tolist()
            num_cols = [c for c in preferred_cols if c in numeric_all]

            if len(num_cols) < 2:
                num_cols = numeric_all

            clean_cols = []
            for c in num_cols:
                if df_filtered[c].std() == 0 or df_filtered[c].isna().all():
                    continue
                clean_cols.append(c)
            num_cols = clean_cols

            if len(num_cols) < 2:
                st.info("상관관계를 계산할 수치형 컬럼이 부족합니다.")
            else:
                st.markdown("##### 📌 분석에 사용된 수치형 변수")
                st.write(", ".join(num_cols))

                corr = df_filtered[num_cols].corr()

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    vmin=-1.0,
                    vmax=1.0,
                    ax=ax,
                )
                ax.set_title("수치형 변수 상관관계 히트맵", pad=16)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("---")
                st.markdown("### 💡 이 상관관계를 보고 생각해볼 수 있는 BM 아이디어")

                insights = []

                def get_corr(a, b):
                    if (a in corr.index) and (b in corr.columns):
                        return corr.loc[a, b]
                    return None

                r_price_ticket = get_corr("Unit price", "avg_ticket")
                if r_price_ticket is not None and r_price_ticket > 0.95:
                    insights.append(
                        "- **단가(Unit price)와 객단가(avg_ticket)가 거의 같이 움직입니다.**  \n"
                        "  → 비싼 상품을 팔수록 한 번에 쓰는 금액도 같이 커진다는 의미입니다.  \n"
                        "  → 고가 상품 라인업을 어떻게 구성할지, 프리미엄 패키지/세트 상품을 만들 수 있을지 고민해 볼 수 있습니다."
                    )

                if "Quantity" in num_cols:
                    weak_targets = []
                    for col in ["Unit price", "Rating"]:
                        r = get_corr("Quantity", col)
                        if r is not None and abs(r) < 0.1:
                            weak_targets.append((col, r))
                    if weak_targets:
                        txt = ", ".join([f"`{c}`(r≈{r:.2f})" for c, r in weak_targets])
                        insights.append(
                            f"- **수량(Quantity)은 {txt} 와(과) 거의 관련이 없습니다.**  \n"
                            "  → 가격을 조금 바꾸거나 평점이 약간 오르내려도, 장바구니에 담는 ‘개수’는 다른 요인에 의해 결정된다는 뜻입니다.  \n"
                            "  → 1+1, 2+1, 묶음 할인 같은 **수량 중심 프로모션 BM**을 따로 설계해 볼 수 있습니다."
                        )

                r_price_total = get_corr("Unit price", "Total")
                r_price_income = get_corr("Unit price", "gross income")
                if (r_price_total is not None and r_price_total >= 0.5) or \
                   (r_price_income is not None and r_price_income >= 0.5):
                    insights.append(
                        "- **단가(Unit price)가 높을수록 매출/이익(Total, gross income)도 커지는 경향이 있습니다.**  \n"
                        "  → 매출을 키우고 싶다면, 단순히 물량만 늘리기보다 **고가·프리미엄 상품의 비중을 어떻게 늘릴지**를 고민해 볼 수 있습니다.  \n"
                        "  → 매장 진열, 추천 상품, 배너 노출에서 고가 라인을 우선 배치하는 전략도 후보가 됩니다."
                    )

                if "Rating" in num_cols:
                    r_rating_total = get_corr("Rating", "Total")
                    if r_rating_total is not None and abs(r_rating_total) < 0.1:
                        insights.append(
                            "- **평점(Rating)과 매출(Total)은 거의 같이 움직이지 않습니다.**  \n"
                            "  → 리뷰 점수가 높다고 해서 매출이 바로 튀어 오르진 않는다는 의미입니다.  \n"
                            "  → 평점은 ‘만족도·브랜딩 관리용 지표’로 두고, 매출은 **가격·프로모션·상품 구성**으로 설계하는 편이 효율적입니다."
                        )

                strong_pairs = []
                for i, c1 in enumerate(num_cols):
                    for c2 in num_cols[i + 1:]:
                        r = corr.loc[c1, c2]
                        if abs(r) >= 0.7:
                            strong_pairs.append((c1, c2, r))

                if strong_pairs:
                    txt = ", ".join(
                        [f"`{a}`-`{b}`(r={r:.2f})" for a, b, r in strong_pairs]
                    )
                    insights.append(
                        f"- **서로 강하게 묶여서 움직이는 지표 조합들**: {txt}  \n"
                        "  → 이 조합들은 한 번에 같이 관리해도 되는 지표들입니다.  \n"
                        "  → 예를 들어 둘 다 거의 같은 모양으로 움직인다면, 대시보드에서 하나는 요약 지표로, 하나는 보조 지표로 두는 식으로 단순화할 수 있습니다."
                    )

                if insights:
                    for line in insights:
                        st.markdown(line)
                else:
                    st.info("이 구간에서는 눈에 띄는 강한 상관/약한 상관 조합이 없습니다. 필터를 바꿔 다른 구간을 살펴볼 수 있습니다.")
