import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Weather Analytics & Forecast", layout="wide")

st.title("🌦️ Weather Analytics & 7-Day Prediction (Optimized)")
st.markdown("⚡ Fast, smooth, and optimized dashboard")

# ---------------- CACHE FUNCTIONS ----------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

@st.cache_resource
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload Weather Dataset", type=["csv"])

if uploaded_file:

    # Load data (cached)
    df = load_data(uploaded_file)

    # Rename columns
    df.rename(columns={
        'Location': 'city',
        'Date_Time': 'date',
        'Temperature_C': 'temperature',
        'Humidity_pct': 'humidity',
        'Precipitation_mm': 'rainfall',
        'Wind_Speed_kmh': 'wind_speed'
    }, inplace=True)

    # Clean data
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(inplace=True)

    # Reduce dataset size (important for speed)
    df = df.tail(3000)

    # Sidebar
    st.sidebar.header("🔍 Filters")
    city = st.sidebar.selectbox("🌍 Select City", df['city'].unique())

    city_df = df[df['city'] == city].copy()

    # Sort once
    city_df = city_df.sort_values('date')

    # ---------------- FAST CHART ----------------
    st.subheader(f"📈 Temperature Trend - {city}")
    st.line_chart(city_df.set_index('date')['temperature'])

    # ---------------- FEATURE ENGINEERING ----------------
    city_df['day_num'] = np.arange(len(city_df))

    X = city_df[['day_num']]
    y = city_df['temperature']

    # Train model (cached)
    model = train_model(X, y)

    # ---------------- PREDICTION ----------------
    future_days = np.arange(len(city_df), len(city_df) + 7).reshape(-1, 1)
    predictions = model.predict(future_days)

    last_date = city_df['date'].max()
    future_dates = pd.date_range(start=last_date, periods=8)[1:]

    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Temperature (°C)': predictions
    })

    # ---------------- OUTPUT ----------------
    st.subheader("🔮 7-Day Forecast")
    st.dataframe(pred_df)

    # ---------------- FAST COMBINED CHART ----------------
    st.subheader("📊 Forecast Visualization")

    chart_df = pd.DataFrame({
        "date": list(city_df['date']) + list(pred_df['Date']),
        "temperature": list(city_df['temperature']) + list(pred_df['Predicted Temperature (°C)'])
    }).set_index("date")

    st.line_chart(chart_df)

    # ---------------- METRICS ----------------
    st.subheader("📊 Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("🌡️ Avg Temp", f"{city_df['temperature'].mean():.2f} °C")
    col2.metric("💧 Avg Humidity", f"{city_df['humidity'].mean():.2f} %")
    col3.metric("🌧️ Avg Rainfall", f"{city_df['rainfall'].mean():.2f} mm")

    # ---------------- EXTREME DETECTION ----------------
    st.subheader("⚠️ Extreme Weather")

    temp_threshold = city_df['temperature'].mean() + 2 * city_df['temperature'].std()
    extreme_days = city_df[city_df['temperature'] > temp_threshold]

    st.success(f"🔥 Extreme Hot Days: {len(extreme_days)}")

else:
    st.info("👆 Upload dataset to start")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚡ Optimized for performance using caching & lightweight charts")
