import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

# 수질 데이터
data = {
    'year': [2020, 2021, 2022, 2023, 2024, 2025],
    'dissolved_oxygen': [15.4, 13.6, 13.9, 12.0, 14.0, 12.3],
    'pH': [7.9, 8.0, 7.2, 7.8, 7.8, 7.3],
    'turbidity': [0.8, 3.6, 1.6, 2.0, 1.6, 1.2]
}

df = pd.DataFrame(data)

# 모델 학습
X = np.array([[y] for y in df['year']])
model_do = LinearRegression().fit(X, df['dissolved_oxygen'])
model_ph = LinearRegression().fit(X, df['pH'])
model_turb = LinearRegression().fit(X, df['turbidity'])

def make_features(year, month):
    return np.array([[year + (month - 1) / 12]])

def predict_water_quality(year, month):
    X_pred = make_features(year, month)
    do = model_do.predict(X_pred)[0]
    ph = model_ph.predict(X_pred)[0]
    turb = model_turb.predict(X_pred)[0]
    return round(do, 2), round(ph, 2), round(turb, 2)

# 등급 분류
def classify_do(do):
    if do >= 8.0:
        return "좋음 💧"
    elif do >= 6.5:
        return "괜찮음 🙂"
    elif do >= 5.0:
        return "살짝 나쁨 😕"
    else:
        return "나쁨 🚫"

def classify_ph(ph):
    if 6.5 <= ph <= 8.5:
        return "좋음 💧"
    elif (6.0 <= ph < 6.5) or (8.5 < ph <= 9.0):
        return "괜찮음 🙂"
    elif (5.5 <= ph < 6.0) or (9.0 < ph <= 9.5):
        return "살짝 나쁨 😕"
    else:
        return "나쁨 🚫"

def classify_turbidity(turb):
    if turb <= 1.0:
        return "좋음 💧"
    elif turb <= 2.0:
        return "괜찮음 🙂"
    elif turb <= 3.0:
        return "살짝 나쁨 😕"
    else:
        return "나쁨 🚫"

# WQI 점수 계산
def calculate_wqi(do, ph, turb):
    # DO: max 20점 (15 이상이면 20점)
    do_score = min(do / 15 * 20, 20)
    
    # pH: 이상적 범위일 때 20점
    if 6.5 <= ph <= 8.5:
        ph_score = 20
    elif 6.0 <= ph < 6.5 or 8.5 < ph <= 9.0:
        ph_score = 15
    elif 5.5 <= ph < 6.0 or 9.0 < ph <= 9.5:
        ph_score = 10
    else:
        ph_score = 5

    # 탁도: 낮을수록 좋음, 0 → 20점, 3 → 0점
    turb_score = max(0, 20 - turb * 6.6)

    # WQI 가중 평균
    wqi = do_score * 0.4 + ph_score * 0.3 + turb_score * 0.3
    return round(wqi, 1)

def classify_wqi(wqi):
    if wqi >= 17:
        return "좋음 💧"
    elif wqi >= 13:
        return "보통 🙂"
    elif wqi >= 10:
        return "주의 😕"
    else:
        return "나쁨 🚫"

# ---------------- Streamlit UI ---------------- #
st.title("🌊 순천 동천 수질 예측 시스템")
st.markdown("날짜를 선택하면 예측 수질과 상태, 그래프, 종합 점수를 보여줍니다.")

# 날짜 입력
selected_date = st.date_input("📅 예측 날짜 선택", value=datetime(2025, 6, 1))

if selected_date:
    year = selected_date.year
    month = selected_date.month

    do, ph, turb = predict_water_quality(year, month)

    status_do = classify_do(do)
    status_ph = classify_ph(ph)
    status_turb = classify_turbidity(turb)

    wqi = calculate_wqi(do, ph, turb)
    wqi_status = classify_wqi(wqi)

    # 결과 출력
    st.subheader(f"📊 {year}년 {month}월 수질 예측 결과:")
    st.markdown(f"- **용존 산소 (DO):** {do} mg/L → {status_do}")
    st.markdown(f"- **pH:** {ph} → {status_ph}")
    st.markdown(f"- **탁도:** {turb} NTU → {status_turb}")
    st.markdown(f"🧮 **종합 수질 점수 (WQI):** {wqi} → {wqi_status}")

    # 시각화: 연도별 예측 그래프
    future_years = list(range(2020, 2031))
    do_preds = [predict_water_quality(y, 6)[0] for y in future_years]
    ph_preds = [predict_water_quality(y, 6)[1] for y in future_years]
    turb_preds = [predict_water_quality(y, 6)[2] for y in future_years]

    graph_df = pd.DataFrame({
        'Year': future_years,
        '용존 산소 (DO)': do_preds,
        'pH': ph_preds,
        '탁도': turb_preds
    }).set_index('Year')

    st.subheader("📈 연도별 수질 예측 그래프 (6월 기준)")
    st.line_chart(graph_df)
