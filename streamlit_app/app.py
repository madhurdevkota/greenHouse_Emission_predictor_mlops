import streamlit as st
import requests
import time
import os
import socket

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Greenhouse Emission Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Title & description
# -------------------------------
st.title("Greenhouse Emission Prediction")
st.markdown(
    """
    <p style="font-size: 18px; color: gray;">
        Predict greenhouse gas emission quantity based on power plant characteristics
    </p>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Fixed domain lists (hardcoded)
# -------------------------------
US_STATES = [
    "Alabama", "Arizona", "Arkansas", "California", "Colorado",
    "Florida", "Georgia", "Illinois", "Indiana", "Kansas",
    "Kentucky", "Louisiana", "Maryland", "Michigan", "Mississippi",
    "Missouri", "Nebraska", "NewJersey", "NewYork", "NorthCarolina",
    "Ohio", "Oklahoma", "Pennsylvania", "SouthCarolina", "Tennessee",
    "Texas", "Virginia", "Washington", "WestVirginia"
]

SOURCE_TYPES = [
    "coal", "gas", "oil", "biomass", "waste", "other_fossil"
]

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2, gap="large")

# -------------------------------
# Input form
# -------------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    capacity = st.slider("Capacity", 50, 5000.0, 250.0, 100.0)
    capacity_factor = st.slider("Capacity Factor", 0.1, 1.0, 0.5, 0.1)
    activity = st.slider("Activity", 50, 16_000_000, 100_000, 10_000 )

    state = st.selectbox("State", options=US_STATES, index=US_STATES.index("Georgia"))
    source_type = st.selectbox("Source Type", options=SOURCE_TYPES, index=0)

    area = st.slider("State Area", 0.001, 70, 50.0,5)
    pop2020 = st.number_input("Population (2020)", min_value=1000, value= 80_000_000, step=10000)

    predict_button = st.button("Predict Emissions", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Results section
# -------------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)

    if predict_button:
        with st.spinner("Calculating prediction..."):

            api_data = {
                "capacity": capacity,
                "capacity_factor": capacity_factor,
                "activity": activity,
                "state": state,
                "source_type": source_type,
                "area": area,
                "pop2020": pop2020
            }

            try:
                api_endpoint = os.getenv("API_URL", "http://model:8000")
                predict_url = f"{api_endpoint.rstrip('/')}/predict"

                response = requests.post(predict_url, json=api_data)
                response.raise_for_status()

                st.session_state.prediction = response.json()
                st.session_state.prediction_time = time.time()

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")
                st.warning("Using mock data for demonstration.")
                st.session_state.prediction = {
                    "predicted_emission": 125000.0,
                    "confidence_interval": [115000.0, 135000.0],
                    "feature_importance": {},
                    "prediction_time": "N/A"
                }

    if "prediction" in st.session_state:
        pred = st.session_state.prediction

        formatted_value = "{:,.2f}".format(pred["predicted_emission"])
        st.markdown(
            f'<div class="prediction-value">{formatted_value}</div>',
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Model Used</p>', unsafe_allow_html=True)
            st.markdown('<p class="info-value">XGBoost</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Prediction Time</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="info-value">{pred["prediction_time"]}</p>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        lower = pred["confidence_interval"][0]
        upper = pred["confidence_interval"][1]

        st.markdown(
            f"<p><strong>Estimated Range:</strong> {lower:,.2f} – {upper:,.2f}</p>",
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            """
            <div style="display: flex; height: 300px; align-items: center;
                        justify-content: center; color: #6b7280; text-align: center;">
                Fill out the form and click "Predict Emissions" to see results.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
version = os.getenv("APP_VERSION", "1.0.0")
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="text-align: center; color: gray;">
        <p><strong>Greenhouse Emission Prediction App</strong></p>
        <p>Version: {version}</p>
        <p>Hostname: {hostname}</p>
        <p>IP Address: {ip_address}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
