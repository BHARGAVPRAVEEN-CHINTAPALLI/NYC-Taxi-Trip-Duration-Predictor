"""
NYC Taxi Trip Duration — Streamlit app

Requires these artifacts (saved by the notebook) in the same folder:
    taxi_ann_model.keras
    taxi_preprocessor.pkl
    taxi_kmeans.pkl
    taxi_config.pkl

Run with:
    streamlit run app_taxi.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="NYC Taxi Trip Duration", page_icon="🚕", layout="centered")


# ----------------------------------------------------------------------
# Load artifacts (cached so it only happens once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("taxi_ann_model.keras")
    preprocessor = joblib.load("taxi_preprocessor.pkl")
    kmeans = joblib.load("taxi_kmeans.pkl")
    config = joblib.load("taxi_config.pkl")
    return model, preprocessor, kmeans, config["columns"]


model, preprocessor, kmeans, FEATURE_COLS = load_artifacts()


# ----------------------------------------------------------------------
# EXACT same feature engineering as the training notebook.
# If these drift from the notebook, predictions become silently wrong.
# ----------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def manhattan(lat1, lon1, lat2, lon2):
    a = haversine(lat1, lon1, lat1, lon2)
    b = haversine(lat1, lon1, lat2, lon1)
    return a + b


def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x))


def add_features(d):
    d = d.copy()
    plat, plon = d["pickup_latitude"], d["pickup_longitude"]
    dlat, dlon = d["dropoff_latitude"], d["dropoff_longitude"]

    d["dist_haversine"] = haversine(plat, plon, dlat, dlon)
    d["dist_manhattan"] = manhattan(plat, plon, dlat, dlon)
    d["bearing"] = bearing(plat, plon, dlat, dlon)

    ts = d["pickup_datetime"]
    d["hour"] = ts.dt.hour
    d["weekday"] = ts.dt.weekday
    d["month"] = ts.dt.month
    d["day_of_year"] = ts.dt.dayofyear
    d["is_weekend"] = (d["weekday"] >= 5).astype(int)
    d["is_rush_hour"] = d["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)

    return d.drop(columns=["pickup_datetime"])


def predict_trip(pickup_dt, plat, plon, dlat, dlon,
                 passenger_count=1, vendor_id=1, store_and_fwd_flag="N"):
    row = pd.DataFrame([{
        "vendor_id": vendor_id,
        "pickup_datetime": pd.to_datetime(pickup_dt),
        "passenger_count": passenger_count,
        "pickup_longitude": plon,
        "pickup_latitude": plat,
        "dropoff_longitude": dlon,
        "dropoff_latitude": dlat,
        "store_and_fwd_flag": store_and_fwd_flag,
    }])

    row = add_features(row)
    row["pickup_zone"] = kmeans.predict(
        row[["pickup_latitude", "pickup_longitude"]].values).astype(str)
    row["dropoff_zone"] = kmeans.predict(
        row[["dropoff_latitude", "dropoff_longitude"]].values).astype(str)

    # align to the exact columns/order the preprocessor was fitted on
    for c in FEATURE_COLS:
        if c not in row.columns:
            row[c] = np.nan
    row = row[FEATURE_COLS]

    prepped = np.asarray(preprocessor.transform(row), dtype="float32")
    pred_log = float(model.predict(prepped, verbose=0)[0][0])
    return float(np.expm1(pred_log)), row   # seconds, and the feature row


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
LANDMARKS = {
    "Times Square":        (40.7580, -73.9855),
    "JFK Airport":         (40.6413, -73.7781),
    "LaGuardia Airport":   (40.7769, -73.8740),
    "Central Park":        (40.7829, -73.9654),
    "Wall Street":         (40.7060, -74.0088),
    "Grand Central":       (40.7527, -73.9772),
    "Brooklyn Bridge":     (40.7061, -73.9969),
    "Empire State Bldg":   (40.7484, -73.9857),
    "Custom (enter below)": None,
}

st.title("🚕 NYC Taxi Trip Duration Predictor")
st.caption("Feedforward neural network · trained on 1.4M NYC taxi trips (RMSLE 0.31, R² 0.81)")


def location_picker(label, default_name, key):
    """Dropdown of landmarks, with a custom lat/lon fallback."""
    choice = st.selectbox(label, list(LANDMARKS.keys()),
                          index=list(LANDMARKS).index(default_name), key=key)
    if LANDMARKS[choice] is None:
        c1, c2 = st.columns(2)
        lat = c1.number_input("latitude", value=40.7580, format="%.4f", key=key + "_lat")
        lon = c2.number_input("longitude", value=-73.9855, format="%.4f", key=key + "_lon")
        return lat, lon
    return LANDMARKS[choice]


col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Pickup")
    p_lat, p_lon = location_picker("From", "Times Square", "pickup")
with col_b:
    st.subheader("Dropoff")
    d_lat, d_lon = location_picker("To", "JFK Airport", "dropoff")

st.divider()

c1, c2, c3 = st.columns(3)
date = c1.date_input("Date", value=pd.Timestamp("2016-03-14"))
time = c2.time_input("Pickup time", value=pd.Timestamp("2016-03-14 17:00").time())
passengers = c3.number_input("Passengers", min_value=1, max_value=6, value=1)

with st.expander("Advanced options"):
    vendor = st.selectbox("Vendor ID", [1, 2], index=0)
    fwd_flag = st.selectbox("store_and_fwd_flag", ["N", "Y"], index=0)

pickup_dt = pd.Timestamp.combine(pd.Timestamp(date).date(), time)

if st.button("Predict trip duration", type="primary"):
    if (p_lat, p_lon) == (d_lat, d_lon):
        st.warning("Pickup and dropoff are the same location.")
    else:
        secs, feat = predict_trip(
            pickup_dt, p_lat, p_lon, d_lat, d_lon,
            passenger_count=passengers, vendor_id=vendor, store_and_fwd_flag=fwd_flag,
        )

        mins = secs / 60
        arrival = pickup_dt + pd.Timedelta(seconds=secs)

        st.success(f"### 🕒 {mins:.1f} minutes  ({secs:.0f} seconds)")

        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated arrival", arrival.strftime("%H:%M"))
        m2.metric("Straight-line distance", f"{feat['dist_haversine'].iloc[0]:.1f} km")
        m3.metric("Rush hour?", "Yes" if feat["is_rush_hour"].iloc[0] == 1 else "No")

        # map of the two points
        st.map(pd.DataFrame({"lat": [p_lat, d_lat], "lon": [p_lon, d_lon]}), zoom=10)

        with st.expander("Features the model actually saw"):
            st.dataframe(feat.T.rename(columns={0: "value"}), use_container_width=True)

        st.caption(
            "Note: distance is straight-line, not road distance — the model learned to "
            "compensate. Typical error is about ±3 minutes (MAE 171s)."
        )
