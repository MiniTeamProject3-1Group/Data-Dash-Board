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
    page_title="이커머스 매출 분석 대시보드",
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
    # Date
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
    needed = {"Invoice ID", "Branch", "City", "Customer type",
              "Gender", "Product line", "Payment", "Rating"}
    has_total = ("Sales" in df.columns) or ("Total" in df.columns)
    return needed.issubset(set(df.columns)) and has_total


# --------------------
# 상단 헤더
# --------------------
st.title("미니프로젝트3_1조")
st.header("이커머스 - 💸 매출 분석 대시보드")
st.write("슈퍼마켓 이커머스 데이터를 기반으로, **월별 매출·고객·상품·시간대 인사이트를 통해 BM 전략**을 세울 수 있도록 만든 대시보드입니다.")

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
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ 파일 업로드 성공!")
        if is_supermarket_schema(df):
            if "Sales" in df.columns and "Total" not in df.columns:
                df = df.rename(columns={"Sales": "Total"})
            df = preprocess_supermarket(df)
            supermarket_mode = True
        else:
            supermarket_mode = False
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

                view_type = st.radio(
                    "분석 단위 선택",
                    ["월별", "일별", "시간대"],
                    horizontal=True
                )

                # 공통: Date 전처리
                df_time = None
                if "Date" in df_filtered.columns:
                    df_time = df_filtered.copy()
                    df_time["Date"] = pd.to_datetime(df_time["Date"], errors="coerce")
                    df_time = df_time.dropna(subset=["Date"])
                    df_time["year_month"] = df_time["Date"].dt.to_period("M").astype(str)

                                    # ===== 월별 분석 =====
                if view_type == "월별":
                    if df_time is None:
                        st.info("월별 분석을 위한 Date 정보가 없습니다.")
                    else:
                        # 1) 어떤 기준들을 쓸지 다중 선택
                        dim_options = ["전체", "Gender", "Customer type", "City", "Product line"]
                        selected_dims = st.multiselect(
                            "월별 세부 구분 기준 (여러 기준을 동시에 선택 가능)",
                            dim_options,
                            default=["전체", "Gender"],  # 기본: 전체 + 성별
                        )

                        if not selected_dims:
                            st.info("적어도 하나의 기준은 선택해야 합니다.")
                        else:
                            # 2) 각 기준 안에서 항목별 다중선택 (Female/Male, 도시들 등)
                            dim_values = {}

                            for dim in dim_options:
                                if dim == "전체" or dim not in selected_dims:
                                    continue
                                if dim not in df_time.columns:
                                    continue

                                candidates = sorted(df_time[dim].dropna().unique())
                                with st.expander(f"{dim} 값 선택", expanded=True):
                                    chosen = st.multiselect(
                                        f"{dim} 값 (여러 개 선택 가능)",
                                        candidates,
                                        default=candidates,
                                        key=f"month_dim_{dim}",
                                    )
                                dim_values[dim] = chosen

                            frames = []

                            # (1) 전체 총 매출 라인
                            if "전체" in selected_dims:
                                overall = (
                                    df_time
                                    .groupby("year_month")["Total"]
                                    .sum()
                                    .reset_index()
                                )
                                overall["series"] = "전체"
                                frames.append(overall[["year_month", "series", "Total"]])

                            # (2) 선택된 각 기준별 라인 (각 기준 안에서도 선택된 항목만)
                            for dim in dim_options:
                                if dim == "전체" or dim not in selected_dims:
                                    continue
                                if dim not in df_time.columns:
                                    continue

                                chosen_vals = dim_values.get(dim, [])
                                if not chosen_vals:
                                    # 그 기준에서 아무 값도 선택 안 했으면 건너뜀
                                    continue

                                df_sub = df_time[df_time[dim].isin(chosen_vals)]

                                tmp = (
                                    df_sub
                                    .groupby(["year_month", dim])["Total"]
                                    .sum()
                                    .reset_index()
                                )
                                # 범례: "Gender: Female", "City: Yangon" 이런 식으로
                                tmp["series"] = tmp[dim].apply(lambda v, d=dim: f"{d}: {v}")
                                frames.append(tmp[["year_month", "series", "Total"]])

                            if not frames:
                                st.info("선택된 기준/값 조합에 해당하는 데이터가 없습니다. 선택을 다시 조정해 주세요.")
                            else:
                                chart_df = pd.concat(frames, ignore_index=True)

                                # x축 월 순서 정렬
                                order = sorted(chart_df["year_month"].unique())
                                chart_df["year_month"] = pd.Categorical(
                                    chart_df["year_month"], categories=order, ordered=True
                                )

                                fig_m = px.line(
                                    chart_df,
                                    x="year_month",
                                    y="Total",
                                    color="series",
                                    markers=True,
                                    title="월별 총 매출 (선택 기준/항목별)",
                                )
                                fig_m.update_layout(
                                    xaxis_title="월",
                                    yaxis_title="총 매출",
                                )
                                st.plotly_chart(fig_m, use_container_width=True)

                # ===== 일별 분석 =====
                elif view_type == "일별":
                    if df_time is None:
                        st.info("일별 분석을 위한 Date 정보가 없습니다.")
                    else:
                        daily = (
                            df_time
                            .groupby("Date")["Total"]
                            .sum()
                            .reset_index()
                        )
                        fig_d = px.line(
                            daily,
                            x="Date",
                            y="Total",
                            markers=True,
                            title="일별 총 매출",
                        )
                        fig_d.update_layout(
                            xaxis_title="날짜",
                            yaxis_title="총 매출",
                        )
                        st.plotly_chart(fig_d, use_container_width=True)

                # ===== 시간대 분석 =====
                else:  # view_type == "시간대"
                    if "period" not in df_filtered.columns:
                        st.info("시간대 분석을 위한 Time 정보가 없습니다.")
                    else:
                        ht = (
                            df_filtered
                            .groupby("period")["Total"]
                            .sum()
                            .reset_index()
                        )
                        order = ["Morning", "Lunch", "Afternoon", "Evening", "Night", "Unknown"]
                        ht["period"] = pd.Categorical(ht["period"], categories=order, ordered=True)
                        ht = ht.sort_values("period")

                        fig_t = px.bar(
                            ht,
                            x="period",
                            y="Total",
                            text_auto=".2s",
                            title="시간대(period)별 총 매출",
                        )
                        fig_t.update_layout(
                            xaxis_title="시간대",
                            yaxis_title="총 매출",
                        )
                        st.plotly_chart(fig_t, use_container_width=True)
        # ===== 상관관계 =====
        with tab_corr:
            st.subheader("📉 수치형 변수 상관관계 분석")

            # 1) 핵심 변수 위주로만 보기
            #   - 계산값 위주(Tax 5%, cogs, gross margin percentage)는 제외
            preferred_cols = [
                "Unit price",       # 개당 가격
                "Quantity",         # 수량
                "Total",            # 총 매출
                "gross income",     # 이익
                "Rating",           # 평점
                "avg_ticket",       # 객단가
            ]

            numeric_all = df_filtered.select_dtypes(include=["float64", "int64"]).columns.tolist()
            # 실제 존재하는 컬럼만 남기기
            num_cols = [c for c in preferred_cols if c in numeric_all]

            # 혹시 모를 fallback
            if len(num_cols) < 2:
                num_cols = numeric_all

            # 분산(표준편차)이 0인 상수 컬럼은 제거 (상관계수 정의 안 됨)
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

                # 2) 상관계수 행렬 계산
                corr = df_filtered[num_cols].corr()

                # 3) 히트맵 시각화 (글자 안 잘리도록 크게 + 라벨 회전)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    vmin=0,
                    vmax=1.0,
                    ax=ax,
                )
                ax.set_title("수치형 변수 상관관계 히트맵", pad=16)

                # 축 라벨 각도/정렬 조정
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

                plt.tight_layout()
                st.pyplot(fig)

                # 4) 자동 BM 인사이트 생성
                st.markdown("---")
                st.markdown("### 💡 상관관계 기반 자동 BM 인사이트")

                insights = []

                def get_corr(a, b):
                    if (a in corr.index) and (b in corr.columns):
                        return corr.loc[a, b]
                    return None

                # (1) Unit price ↔ avg_ticket
                r_price_ticket = get_corr("Unit price", "avg_ticket")
                if r_price_ticket is not None and r_price_ticket > 0.95:
                    insights.append(
                        "- `Unit price`(단가)와 `avg_ticket`(객단가)의 상관계수가 **0.95 이상**입니다. "
                        "→ 두 변수는 정보가 거의 동일하므로, 모델링/대시보드에서는 **둘 중 하나만 대표 변수로 사용**해도 됩니다."
                    )

                # (2) Quantity가 Unit price & Rating과 약한 상관
                if "Quantity" in num_cols:
                    weak_targets = []
                    for col in ["Unit price", "Rating"]:
                        r = get_corr("Quantity", col)
                        if r is not None and abs(r) < 0.1:
                            weak_targets.append((col, r))
                    if weak_targets:
                        txt = ", ".join([f"`{c}`(r≈{r:.2f})" for c, r in weak_targets])
                        insights.append(
                            f"- `Quantity`(수량)은 {txt} 와(과) 상관계수가 거의 0에 가깝습니다. "
                            "→ 가격이나 평점이 바뀌어도 **구매 수량은 별도의 요인(상품 특성, 프로모션 등)에 의해 결정**된다는 의미입니다. "
                            "묶음 할인·1+1 같은 **수량 기반 프로모션 BM**을 따로 설계할 여지가 있습니다."
                        )

                # (3) 높은 단가 ↔ 높은 매출/이익
                r_price_total = get_corr("Unit price", "Total")
                r_price_income = get_corr("Unit price", "gross income")
                if (r_price_total is not None and r_price_total >= 0.5) or \
                   (r_price_income is not None and r_price_income >= 0.5):
                    insights.append(
                        "- `Unit price`(단가)와 `Total`/`gross income`(매출/이익) 간 상관계수가 **0.5 이상**으로 꽤 높습니다. "
                        "→ **고가 상품일수록 매출·이익 기여도가 크다**는 뜻이며, "
                        "고가 상품 라인에 노출·마케팅·재고를 우선 배치하는 BM 전략이 유효합니다."
                    )

                # (4) Rating ↔ Total
                if "Rating" in num_cols:
                    r_rating_total = get_corr("Rating", "Total")
                    if r_rating_total is not None and abs(r_rating_total) < 0.1:
                        insights.append(
                            "- `Rating`(평점)은 `Total`(매출)과 상관계수가 거의 0입니다. "
                            "→ 리뷰 평점이 높다고 해서 곧바로 매출이 올라간다는 근거는 약합니다. "
                            "평점은 **만족도/브랜딩 지표**로 활용하고, 매출 증대는 **가격·프로모션·상품구성**으로 설계하는 게 더 효율적입니다."
                        )

                # (5) 강한 상관 요약 (중복 정보 체크용)
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
                        f"- 강한 상관( |r| ≥ 0.7 )을 보이는 조합: {txt}  \n"
                        "  → 이 조합들은 **함께 움직이는 지표**이므로, BM 설계 시 한 묶음으로 보거나 중복 여부를 검토할 수 있습니다."
                    )

                if insights:
                    for line in insights:
                        st.markdown(line)
                else:
                    st.info("눈에 띄게 강한 상관/약한 상관 조합은 없습니다. 필터를 조정해 다른 구간을 살펴볼 수 있습니다.")
