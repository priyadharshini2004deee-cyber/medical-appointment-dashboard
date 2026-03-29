
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.over_sampling import SMOTE

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Medical Appointment Advanced Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# STYLES
# =========================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 100%);
}
.block-container {
    max-width: 1650px;
    padding-top: 1.9rem;
    padding-bottom: 1rem;
}
.main-title {
    font-size: 2.05rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 0.15rem;
    margin-bottom: 0.35rem;
    line-height: 1.28;
    white-space: normal;
    overflow-wrap: anywhere;
}
.sub-title {
    color: #475569;
    font-size: 0.98rem;
    margin-bottom: 1.15rem;
    line-height: 1.55;
    white-space: normal;
    overflow-wrap: anywhere;
}
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
    border: 1px solid #dbeafe;
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1d4ed8 100%);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    border: none;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white;
    font-weight: 700;
}
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e3a8a;
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 12px;
}
.good-box {
    background: #dcfce7;
    border: 1px solid #86efac;
    color: #166534;
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 12px;
}
.avg-box {
    background: #fef9c3;
    border: 1px solid #fde68a;
    color: #854d0e;
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 12px;
}
.weak-box {
    background: #fee2e2;
    border: 1px solid #fca5a5;
    color: #991b1b;
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 12px;
}
.alert-box {
    background: #fff7ed;
    border: 1px solid #fdba74;
    color: #9a3412;
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 12px;
}
.small-note {
    color: #64748b;
    font-size: 0.92rem;
}
h2, h3 {
    color: #0f172a;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Medical_appointment_data.csv"

# =========================================================
# HELPERS
# =========================================================
def clean_text(series):
    s = series.astype(str).str.strip()
    s = s.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    return s.fillna("Unknown")

def safe_multiselect(label, options, default=None, key=None):
    options = [str(x) for x in options if pd.notna(x) and str(x).strip() != ""]
    options = sorted(list(dict.fromkeys(options)))
    if not options:
        return []
    if default is None:
        default = options
    default = [str(x) for x in default if str(x) in options]
    if not default:
        default = options
    return st.sidebar.multiselect(label, options, default=default, key=key)

def infer_binary_target(series):
    s = series.astype(str).str.strip().str.lower()
    return s.map({
        "yes": 1, "y": 1, "1": 1, "true": 1, "no-show": 1,
        "no": 0, "n": 0, "0": 0, "false": 0, "attended": 0
    })

def create_patient_alerts(dataframe):
    temp = dataframe.copy()
    if temp.empty:
        return pd.DataFrame()

    alert_df = temp.groupby("patient_id_final").agg(
        total_appointments=("patient_id_final", "size"),
        no_show_count=("no_show_label", lambda x: (x == "No-Show").sum()),
        attended_count=("no_show_label", lambda x: (x == "Attended").sum()),
        avg_temp=("average_temp_day", "mean") if "average_temp_day" in temp.columns else ("patient_id_final", "size"),
        avg_rain=("average_rain_day", "mean") if "average_rain_day" in temp.columns else ("patient_id_final", "size"),
        latest_date=("appointment_date", "max") if "appointment_date" in temp.columns else ("patient_id_final", "size")
    ).reset_index()

    alert_df["no_show_rate"] = np.where(
        alert_df["total_appointments"] > 0,
        alert_df["no_show_count"] / alert_df["total_appointments"] * 100,
        0
    )

    def alert_label(row):
        total = row["total_appointments"]
        missed = row["no_show_count"]
        rate = row["no_show_rate"]
        if total >= 3 and missed >= 2 and rate >= 60:
            return "High Alert"
        elif total >= 2 and missed >= 1 and rate >= 50:
            return "Medium Alert"
        elif total == 1 and missed == 1:
            return "Medium Alert"
        else:
            return "Low Alert"

    alert_df["alert_level"] = alert_df.apply(alert_label, axis=1)
    order_map = {"High Alert": 0, "Medium Alert": 1, "Low Alert": 2}
    alert_df["alert_order"] = alert_df["alert_level"].map(order_map)
    alert_df = alert_df.sort_values(
        ["alert_order", "no_show_rate", "total_appointments"],
        ascending=[True, False, False]
    ).drop(columns=["alert_order"])
    return alert_df

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error("Medical_appointment_data.csv file not found.")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {
        "appointment_date_continuous": "appointment_date",
        "no_show": "no_show",
        "place": "place"
    }
    df = df.rename(columns=rename_map)

    if "appointment_date" in df.columns:
        df["appointment_date"] = pd.to_datetime(df["appointment_date"], errors="coerce")
        df["weekday"] = df["appointment_date"].dt.day_name()
        df["month"] = df["appointment_date"].dt.month
        df["year"] = df["appointment_date"].dt.year
        df["day"] = df["appointment_date"].dt.day
    else:
        df["appointment_date"] = pd.NaT
        df["weekday"] = "Unknown"
        df["month"] = 0
        df["year"] = 0
        df["day"] = 0

    actual_patient_id_col = None
    for col in ["patientid", "patient_id", "patientid_"]:
        if col in df.columns:
            actual_patient_id_col = col
            break

    if actual_patient_id_col is not None:
        df["patient_id_final"] = clean_text(df[actual_patient_id_col])
        has_real_patient_id = True
    else:
        df["patient_id_final"] = [f"REC{i:06d}" for i in range(1, len(df) + 1)]
        has_real_patient_id = False

    text_cols = [
        "gender", "specialty", "disability", "place",
        "appointment_shift", "rain_intensity", "heat_intensity", "weekday"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = clean_text(df[col])

    if "no_show" in df.columns:
        target_num = infer_binary_target(df["no_show"])
        df["no_show_target"] = target_num
        df["no_show_label"] = np.where(df["no_show_target"] == 1, "No-Show", "Attended")
        df.loc[df["no_show_target"].isna(), "no_show_label"] = "Unknown"
    else:
        df["no_show_target"] = np.nan
        df["no_show_label"] = "Unknown"

    numeric_cols = [
        "age", "appointment_time", "under_12_years_old", "over_60_years_old",
        "patient_needs_companion", "average_temp_day", "average_rain_day",
        "max_temp_day", "max_rain_day", "rainy_day_before", "storm_day_before",
        "hipertension", "diabetes", "alcoholism", "handcap",
        "scholarship", "sms_received"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    meta = {
        "has_real_patient_id": has_real_patient_id,
        "actual_patient_id_col": actual_patient_id_col
    }
    return df, meta

df, meta = load_data()

# =========================================================
# TRAINING / FORECAST HELPERS
# =========================================================
@st.cache_data
def train_classification_bundle(dataframe):
    model_df = dataframe.copy().dropna(subset=["no_show_target"])

    feature_cols_num = [
        "age", "sms_received", "hipertension", "diabetes", "alcoholism", "handcap",
        "average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day",
        "rainy_day_before", "storm_day_before", "under_12_years_old",
        "over_60_years_old", "patient_needs_companion", "appointment_time"
    ]
    feature_cols_cat = [
        "gender", "appointment_shift", "weekday", "rain_intensity", "heat_intensity"
    ]

    feature_cols_num = [c for c in feature_cols_num if c in model_df.columns]
    feature_cols_cat = [c for c in feature_cols_cat if c in model_df.columns]

    X_num = model_df[feature_cols_num].copy() if feature_cols_num else pd.DataFrame(index=model_df.index)
    X_num = X_num.fillna(X_num.median(numeric_only=True)).fillna(0)

    if feature_cols_cat:
        X_cat = pd.get_dummies(model_df[feature_cols_cat].fillna("Unknown"), drop_first=False)
    else:
        X_cat = pd.DataFrame(index=model_df.index)

    X = pd.concat([X_num, X_cat], axis=1)
    y = model_df["no_show_target"].astype(int)

    if y.nunique() < 2 or len(X) < 20:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    before_counts = y_train.value_counts().to_dict()

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    after_counts = pd.Series(y_train_sm).value_counts().to_dict()

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_sm, y_train_sm)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": clf.feature_importances_
    }).sort_values("Importance", ascending=False)

    return {
        "model": clf,
        "features": X.columns.tolist(),
        "numeric_features": feature_cols_num,
        "X_reference": X,
        "acc": acc,
        "f1": f1,
        "roc": roc,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "tpr": tpr,
        "cm": cm,
        "importance_df": importance_df,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "train_before": len(X_train),
        "train_after": len(X_train_sm),
        "test_rows": len(X_test),
        "original_rows": len(X)
    }

@st.cache_data
def build_forecast_table(dataframe, forecast_days, start_date):
    temp = dataframe.copy().dropna(subset=["appointment_date"])
    if temp.empty:
        return pd.DataFrame()

    daily = temp.groupby(temp["appointment_date"].dt.date).agg(
        appointments=("appointment_date", "size"),
        avg_temp=("average_temp_day", "mean") if "average_temp_day" in temp.columns else ("appointment_date", "size"),
        avg_rain=("average_rain_day", "mean") if "average_rain_day" in temp.columns else ("appointment_date", "size")
    ).reset_index()

    daily.columns = ["date", "appointments", "avg_temp", "avg_rain"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    if len(daily) < 10:
        return pd.DataFrame()

    daily["avg_temp"] = pd.to_numeric(daily["avg_temp"], errors="coerce").fillna(0)
    daily["avg_rain"] = pd.to_numeric(daily["avg_rain"], errors="coerce").fillna(0)
    daily["day"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    daily["weekday"] = daily["date"].dt.dayofweek

    Xf = daily[["day", "month", "year", "weekday", "avg_temp", "avg_rain"]]
    yf = daily["appointments"]

    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(Xf, yf)

    future_dates = pd.date_range(start=pd.to_datetime(start_date), periods=forecast_days)
    temp_median = float(daily["avg_temp"].median())
    rain_median = float(daily["avg_rain"].median())

    forecast_rows = []
    for d in future_dates:
        row = pd.DataFrame([{
            "day": d.day,
            "month": d.month,
            "year": d.year,
            "weekday": d.dayofweek,
            "avg_temp": temp_median,
            "avg_rain": rain_median
        }])
        pred = max(0, int(round(reg.predict(row)[0])))
        forecast_rows.append({"Date": d.date(), "Expected Patients": pred})

    return pd.DataFrame(forecast_rows)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🏥 Medical Appointment Advanced Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Overview, Patient Explorer, Weather Analysis, Specialty Analysis, Place Analysis, Prediction, Forecast, Time Series, Train-Test Metrics, Patient Alerts, Date-wise Analysis, and Data Explorer.</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🔎 Filters")

gender_options = sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else []
place_options = sorted(df["place"].dropna().unique().tolist()) if "place" in df.columns else []
weekday_options = sorted(df["weekday"].dropna().unique().tolist()) if "weekday" in df.columns else []
specialty_options = sorted(df["specialty"].dropna().unique().tolist()) if "specialty" in df.columns else []
shift_options = sorted(df["appointment_shift"].dropna().unique().tolist()) if "appointment_shift" in df.columns else []
status_options = sorted(df["no_show_label"].dropna().unique().tolist()) if "no_show_label" in df.columns else []

selected_page = st.sidebar.radio(
    "Go to Section",
    [
        "Overview", "Patient Explorer", "Weather Analysis", "Specialty Analysis",
        "Place Analysis", "Prediction", "Forecast", "Time Series",
        "Train-Test Metrics", "Patient Alerts", "Date-wise Analysis", "Data Explorer"
    ]
)

selected_gender = safe_multiselect("Gender", gender_options, gender_options, "gender_filter")
selected_place = safe_multiselect("Place", place_options, place_options[:15] if len(place_options) > 15 else place_options, "place_filter")
selected_weekday = safe_multiselect("Weekday", weekday_options, weekday_options, "weekday_filter")
selected_specialty = safe_multiselect("Specialty", specialty_options, specialty_options, "specialty_filter")
selected_shift = safe_multiselect("Appointment Shift", shift_options, shift_options, "shift_filter")
selected_status = safe_multiselect("Attendance Status", status_options, status_options, "status_filter")

age_min = int(df["age"].min()) if "age" in df.columns and df["age"].notna().any() else 0
age_max = int(df["age"].max()) if "age" in df.columns and df["age"].notna().any() else 100
selected_age = st.sidebar.slider("Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

valid_dates = df["appointment_date"].dropna() if "appointment_date" in df.columns else pd.Series(dtype="datetime64[ns]")
if not valid_dates.empty:
    selected_dates = st.sidebar.date_input(
        "Appointment Date Range",
        value=(valid_dates.min().date(), valid_dates.max().date())
    )
else:
    selected_dates = None

# =========================================================
# FILTER DATA
# =========================================================
filtered = df.copy()
if selected_gender and "gender" in filtered.columns:
    filtered = filtered[filtered["gender"].isin(selected_gender)]
if selected_place and "place" in filtered.columns:
    filtered = filtered[filtered["place"].isin(selected_place)]
if selected_weekday and "weekday" in filtered.columns:
    filtered = filtered[filtered["weekday"].isin(selected_weekday)]
if selected_specialty and "specialty" in filtered.columns:
    filtered = filtered[filtered["specialty"].isin(selected_specialty)]
if selected_shift and "appointment_shift" in filtered.columns:
    filtered = filtered[filtered["appointment_shift"].isin(selected_shift)]
if selected_status and "no_show_label" in filtered.columns:
    filtered = filtered[filtered["no_show_label"].isin(selected_status)]
if "age" in filtered.columns:
    filtered = filtered[filtered["age"].fillna(-1).between(selected_age[0], selected_age[1])]

if selected_dates and isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2:
    start_date = pd.to_datetime(selected_dates[0])
    end_date = pd.to_datetime(selected_dates[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if "appointment_date" in filtered.columns:
        filtered = filtered[filtered["appointment_date"].between(start_date, end_date)]

if filtered.empty:
    st.warning("No data available for the selected filters. Please widen the filters.")
    st.stop()

# =========================================================
# COMMON METRICS
# =========================================================
total_appointments = len(filtered)
total_dataset_records = len(df)

if meta["has_real_patient_id"]:
    unique_patients_display = int(filtered["patient_id_final"].nunique())
    total_unique_patients_full = int(df["patient_id_final"].nunique())
    patient_note = f"Real patient ID found: {meta['actual_patient_id_col']}"
else:
    unique_patients_display = None
    total_unique_patients_full = None
    patient_note = "Real patient ID not found in dataset. Dashboard uses record-based IDs for explorer."

attended_count = int((filtered["no_show_label"] == "Attended").sum())
noshow_count = int((filtered["no_show_label"] == "No-Show").sum())
noshow_rate = (noshow_count / total_appointments * 100) if total_appointments > 0 else 0
avg_age = float(filtered["age"].mean()) if "age" in filtered.columns and filtered["age"].notna().any() else 0.0
top_specialty = filtered["specialty"].mode().iloc[0] if "specialty" in filtered.columns and filtered["specialty"].notna().any() else "N/A"

bundle = train_classification_bundle(filtered)
patient_alerts_df = create_patient_alerts(filtered)

# =========================================================
# PAGES
# =========================================================
if selected_page == "Overview":
    st.markdown("## 📊 Overview")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Filtered Appointments", f"{total_appointments:,}")
    m2.metric("Total Dataset Records", f"{total_dataset_records:,}")
    m3.metric("Unique Patients", f"{unique_patients_display:,}" if unique_patients_display is not None else "N/A")
    m4.metric("Attended", f"{attended_count:,}")
    m5.metric("No-Show", f"{noshow_count:,}")
    m6.metric("No-Show Rate", f"{noshow_rate:.2f}%")

    st.markdown(f'<div class="info-box">{patient_note}</div>', unsafe_allow_html=True)
    if total_unique_patients_full is not None:
        st.markdown(f'<div class="info-box">Total unique patients in full dataset: <b>{total_unique_patients_full:,}</b></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        status_df = filtered["no_show_label"].value_counts().reset_index()
        status_df.columns = ["Status", "Count"]
        fig = px.pie(status_df, names="Status", values="Count", hole=0.5, title="Attendance Status")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "specialty" in filtered.columns:
            sp_df = filtered["specialty"].value_counts().reset_index().head(10)
            sp_df.columns = ["Specialty", "Appointments"]
            fig = px.bar(sp_df, x="Appointments", y="Specialty", orientation="h", title="Top Specialties")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if "place" in filtered.columns:
            place_df = filtered["place"].value_counts().reset_index().head(10)
            place_df.columns = ["Place", "Appointments"]
            fig = px.bar(place_df, x="Appointments", y="Place", orientation="h", title="Top Places")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        if "appointment_date" in filtered.columns and filtered["appointment_date"].notna().any():
            daily = filtered.groupby(filtered["appointment_date"].dt.date).size().reset_index(name="Appointments")
            daily.columns = ["Date", "Appointments"]
            fig = px.line(daily, x="Date", y="Appointments", markers=True, title="Appointment Trend")
            st.plotly_chart(fig, use_container_width=True)

elif selected_page == "Patient Explorer":
    st.markdown("## 👤 Patient Explorer")
    st.markdown(f'<div class="info-box">{patient_note}</div>', unsafe_allow_html=True)

    patient_list = sorted(filtered["patient_id_final"].astype(str).unique().tolist())
    selected_patient = st.selectbox("Select Patient ID", patient_list)
    patient_df = filtered[filtered["patient_id_final"].astype(str) == selected_patient].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Patient Appointments", len(patient_df))
    c2.metric("Patient Attended", int((patient_df["no_show_label"] == "Attended").sum()))
    c3.metric("Patient No-Show", int((patient_df["no_show_label"] == "No-Show").sum()))

    show_cols = [c for c in [
        "patient_id_final", "appointment_date", "specialty", "appointment_shift", "gender",
        "age", "no_show_label", "place", "disability", "rain_intensity",
        "heat_intensity", "sms_received", "average_temp_day", "average_rain_day"
    ] if c in patient_df.columns]

    st.dataframe(
        patient_df[show_cols].sort_values(by="appointment_date", ascending=False) if "appointment_date" in patient_df.columns else patient_df[show_cols],
        use_container_width=True
    )

elif selected_page == "Weather Analysis":
    st.markdown("## 🌦️ Weather Analysis")
    c1, c2 = st.columns(2)
    with c1:
        if "rain_intensity" in filtered.columns:
            rain_df = filtered.groupby(["rain_intensity", "no_show_label"]).size().reset_index(name="Count")
            fig = px.bar(rain_df, x="rain_intensity", y="Count", color="no_show_label", barmode="group", title="Rain Intensity vs Attendance")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "heat_intensity" in filtered.columns:
            heat_df = filtered.groupby(["heat_intensity", "no_show_label"]).size().reset_index(name="Count")
            fig = px.bar(heat_df, x="heat_intensity", y="Count", color="no_show_label", barmode="group", title="Heat Intensity vs Attendance")
            st.plotly_chart(fig, use_container_width=True)

    weather_cols = [c for c in ["average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day"] if c in filtered.columns]
    if weather_cols:
        summary = filtered.groupby("no_show_label")[weather_cols].mean().reset_index()
        st.dataframe(summary, use_container_width=True)

elif selected_page == "Specialty Analysis":
    st.markdown("## 🩺 Specialty Analysis")
    sp_df = filtered["specialty"].value_counts().reset_index().head(12)
    sp_df.columns = ["Specialty", "Appointments"]
    fig = px.bar(sp_df, x="Appointments", y="Specialty", orientation="h", title="Top Specialties by Appointments")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    if "specialty" in filtered.columns:
        summary_df = filtered.groupby("specialty").agg(
            appointments=("specialty", "size"),
            avg_age=("age", "mean"),
            no_show_count=("no_show_label", lambda x: (x == "No-Show").sum())
        ).reset_index()
        summary_df["no_show_rate"] = np.where(summary_df["appointments"] > 0, summary_df["no_show_count"] / summary_df["appointments"] * 100, 0)
        st.dataframe(summary_df.sort_values("appointments", ascending=False), use_container_width=True)

elif selected_page == "Place Analysis":
    st.markdown("## 🗺️ Place Analysis")
    place_df = filtered["place"].value_counts().reset_index().head(15)
    place_df.columns = ["Place", "Appointments"]
    fig = px.bar(place_df, x="Appointments", y="Place", orientation="h", title="Top Places by Appointments")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    place_summary = filtered.groupby("place").agg(
        appointments=("place", "size"),
        unique_patients=("patient_id_final", "nunique"),
        no_show_count=("no_show_label", lambda x: (x == "No-Show").sum())
    ).reset_index()
    place_summary["no_show_rate"] = np.where(place_summary["appointments"] > 0, place_summary["no_show_count"] / place_summary["appointments"] * 100, 0)
    st.dataframe(place_summary.sort_values("appointments", ascending=False), use_container_width=True)

elif selected_page == "Prediction":
    st.markdown("## 🤖 Prediction")

    if bundle is None:
        st.warning("Not enough valid data to train prediction model.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        input_age = c1.number_input("Age", min_value=0, max_value=120, value=30)
        input_gender = c2.selectbox("Gender", gender_options if gender_options else ["F", "M"])
        input_sms = c3.selectbox("SMS Received", [0, 1], index=0)
        input_shift = c4.selectbox("Appointment Shift", shift_options if shift_options else ["Morning", "Afternoon"])

        c5, c6, c7, c8 = st.columns(4)
        input_temp = c5.number_input("Average Temperature", value=30.0)
        input_rain = c6.number_input("Average Rain", value=0.0)
        input_rainy_before = c7.selectbox("Rainy Day Before", [0, 1], index=0)
        input_storm_before = c8.selectbox("Storm Day Before", [0, 1], index=0)

        c9, c10, c11, c12 = st.columns(4)
        input_weekday = c9.selectbox("Weekday", weekday_options if weekday_options else ["Monday"])
        input_heat = c10.selectbox("Heat Intensity", sorted(df["heat_intensity"].dropna().unique().tolist()) if "heat_intensity" in df.columns else ["Unknown"])
        input_rain_intensity = c11.selectbox("Rain Intensity", sorted(df["rain_intensity"].dropna().unique().tolist()) if "rain_intensity" in df.columns else ["Unknown"])
        input_time = c12.number_input("Appointment Time", value=10.0)

        if st.button("Predict No-Show"):
            features = bundle["features"]
            input_row = {col: 0 for col in features}

            numeric_defaults = {
                "age": input_age,
                "sms_received": input_sms,
                "average_temp_day": input_temp,
                "average_rain_day": input_rain,
                "rainy_day_before": input_rainy_before,
                "storm_day_before": input_storm_before,
                "appointment_time": input_time,
            }
            for col, val in numeric_defaults.items():
                if col in input_row:
                    input_row[col] = val

            for col in bundle["numeric_features"]:
                if col in input_row and input_row[col] == 0:
                    try:
                        input_row[col] = float(bundle["X_reference"][col].median())
                    except Exception:
                        input_row[col] = 0

            cat_keys = [
                f"gender_{input_gender}",
                f"appointment_shift_{input_shift}",
                f"weekday_{input_weekday}",
                f"heat_intensity_{input_heat}",
                f"rain_intensity_{input_rain_intensity}",
            ]
            for key in cat_keys:
                if key in input_row:
                    input_row[key] = 1

            pred_df = pd.DataFrame([input_row])[features]
            pred = bundle["model"].predict(pred_df)[0]
            prob = float(bundle["model"].predict_proba(pred_df)[0][1] * 100)

            if pred == 1:
                st.error(f"Prediction: Likely No-Show ({prob:.2f}% probability)")
            else:
                st.success(f"Prediction: Likely Attend ({100 - prob:.2f}% confidence)")

elif selected_page == "Forecast":
    st.markdown("## 📅 Forecast")
    forecast_source = st.radio("Forecast Source", ["Use Full Data", "Use Filtered Data"], horizontal=True)
    forecast_base = df if forecast_source == "Use Full Data" else filtered

    if "appointment_date" not in forecast_base.columns or forecast_base["appointment_date"].dropna().empty:
        st.warning("No appointment date data available for forecasting.")
    else:
        last_date = forecast_base["appointment_date"].max()
        fc1, fc2 = st.columns(2)
        manual_start_date = fc1.date_input("Forecast Start Date", value=(last_date + pd.Timedelta(days=1)).date())
        forecast_days = fc2.slider("Select Future Days", 2, 7, 3)

        forecast_df = build_forecast_table(forecast_base, forecast_days, manual_start_date)

        if forecast_df.empty:
            st.warning("Not enough data to build forecast. Use broader filters or full data.")
        else:
            fig = px.line(forecast_df, x="Date", y="Expected Patients", markers=True, title="Future Patient Forecast")
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig2 = px.bar(forecast_df, x="Date", y="Expected Patients", title="Forecast Bar View")
                st.plotly_chart(fig2, use_container_width=True)
            with c2:
                st.dataframe(forecast_df, use_container_width=True)

elif selected_page == "Time Series":
    st.markdown("## 📈 Time Series")
    if "appointment_date" not in filtered.columns or filtered["appointment_date"].dropna().empty:
        st.warning("No date data available.")
    else:
        daily = filtered.groupby(filtered["appointment_date"].dt.date).size().reset_index(name="Appointments")
        daily.columns = ["Date", "Appointments"]

        fig1 = px.line(daily, x="Date", y="Appointments", markers=True, title="Daily Appointments Trend")
        st.plotly_chart(fig1, use_container_width=True)

        daily["Rolling_7"] = daily["Appointments"].rolling(7, min_periods=1).mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=daily["Date"], y=daily["Appointments"], mode="lines+markers", name="Daily"))
        fig2.add_trace(go.Scatter(x=daily["Date"], y=daily["Rolling_7"], mode="lines", name="7-Day Average"))
        fig2.update_layout(title="Daily Appointments with 7-Day Moving Average", xaxis_title="Date", yaxis_title="Appointments")
        st.plotly_chart(fig2, use_container_width=True)

elif selected_page == "Train-Test Metrics":
    st.markdown("## 🧪 Train / Test Split + Metrics")

    if bundle is None:
        st.warning("Not enough valid data to train model.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Original Rows Used", f"{bundle['original_rows']:,}")
        c2.metric("Train Before SMOTE", f"{bundle['train_before']:,}")
        c3.metric("Train After SMOTE", f"{bundle['train_after']:,}")
        c4.metric("Test Rows", f"{bundle['test_rows']:,}")
        c5.metric("Features Used", f"{len(bundle['features'])}")

        if bundle["original_rows"] < 200:
            st.markdown('<div class="weak-box">Model is weak because filtered training data is too small. Use broader filters for better results.</div>', unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{bundle['acc']:.3f}")
        m2.metric("F1 Score", f"{bundle['f1']:.3f}")
        m3.metric("ROC AUC", f"{bundle['roc']:.3f}")
        m4.metric("Precision", f"{bundle['precision']:.3f}")
        m5.metric("Recall", f"{bundle['recall']:.3f}")

        if bundle["f1"] >= 0.75 and bundle["roc"] >= 0.80:
            st.markdown('<div class="good-box">Model is GOOD.</div>', unsafe_allow_html=True)
        elif bundle["f1"] >= 0.60 and bundle["roc"] >= 0.70:
            st.markdown('<div class="avg-box">Model is AVERAGE.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="weak-box">Model is WEAK. Needs improvement.</div>', unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            cm_df = pd.DataFrame(
                bundle["cm"],
                index=["Actual Attended", "Actual No-Show"],
                columns=["Predicted Attended", "Predicted No-Show"]
            )
            fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

        with cc2:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=bundle["fpr"], y=bundle["tpr"], mode="lines",
                name=f"ROC Curve (AUC={bundle['roc']:.3f})"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random", line=dict(dash="dash")
            ))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)

        fig_imp = px.bar(
            bundle["importance_df"].head(15),
            x="Importance", y="Feature", orientation="h",
            title="Top 15 Important Features"
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

elif selected_page == "Patient Alerts":
    st.markdown("## 🚨 Patient Alerts")

    if patient_alerts_df.empty:
        st.warning("No patient alerts available.")
    else:
        high_alert = patient_alerts_df[patient_alerts_df["alert_level"] == "High Alert"]
        medium_alert = patient_alerts_df[patient_alerts_df["alert_level"] == "Medium Alert"]

        c1, c2, c3 = st.columns(3)
        c1.metric("High Alert Patients", len(high_alert))
        c2.metric("Medium Alert Patients", len(medium_alert))
        c3.metric("Total Patients in View", patient_alerts_df["patient_id_final"].nunique())

        st.markdown('<div class="alert-box">Patients are flagged based on repeated no-show behavior and risky no-show pattern.</div>', unsafe_allow_html=True)

        fig = px.bar(
            patient_alerts_df.head(20),
            x="patient_id_final",
            y="no_show_rate",
            color="alert_level",
            title="Top Patient Alert Levels"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(patient_alerts_df, use_container_width=True)

elif selected_page == "Date-wise Analysis":
    st.markdown("## 📅 Date-wise Analysis")

    if "appointment_date" not in filtered.columns or filtered["appointment_date"].dropna().empty:
        st.warning("No date data available.")
    else:
        date_summary = filtered.groupby(filtered["appointment_date"].dt.date).agg(
            total_appointments=("appointment_date", "size"),
            attended=("no_show_label", lambda x: (x == "Attended").sum()),
            no_show=("no_show_label", lambda x: (x == "No-Show").sum()),
            avg_temp=("average_temp_day", "mean") if "average_temp_day" in filtered.columns else ("appointment_date", "size"),
            avg_rain=("average_rain_day", "mean") if "average_rain_day" in filtered.columns else ("appointment_date", "size")
        ).reset_index()

        date_summary.columns = ["Date", "Total Appointments", "Attended", "No-Show", "Average Temp", "Average Rain"]
        date_summary["No-Show Rate"] = np.where(
            date_summary["Total Appointments"] > 0,
            date_summary["No-Show"] / date_summary["Total Appointments"] * 100,
            0
        )

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.line(date_summary, x="Date", y="Total Appointments", markers=True, title="Date-wise Appointments")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.line(date_summary, x="Date", y="No-Show Rate", markers=True, title="Date-wise No-Show Rate")
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig3 = px.bar(date_summary, x="Date", y="Average Rain", title="Date-wise Rain")
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            fig4 = px.bar(date_summary, x="Date", y="Average Temp", title="Date-wise Temperature")
            st.plotly_chart(fig4, use_container_width=True)

        st.dataframe(date_summary, use_container_width=True)

elif selected_page == "Data Explorer":
    st.markdown("## 📋 Data Explorer")
    st.markdown(f'<div class="info-box">{patient_note}</div>', unsafe_allow_html=True)

    show_cols = [c for c in [
        "patient_id_final", "appointment_date", "specialty", "appointment_shift",
        "gender", "age", "no_show_label", "place", "disability",
        "rain_intensity", "heat_intensity", "average_temp_day",
        "average_rain_day", "sms_received", "rainy_day_before", "storm_day_before"
    ] if c in filtered.columns]

    st.dataframe(filtered[show_cols].head(3000), use_container_width=True)

    s1, s2 = st.columns(2)
    with s1:
        if "specialty" in filtered.columns:
            specialty_counts = filtered["specialty"].value_counts().reset_index()
            specialty_counts.columns = ["Specialty", "Count"]
            st.dataframe(specialty_counts, use_container_width=True)
    with s2:
        if "place" in filtered.columns:
            place_counts = filtered["place"].value_counts().reset_index()
            place_counts.columns = ["Place", "Count"]
            st.dataframe(place_counts, use_container_width=True)

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv_data,
        file_name="filtered_medical_appointments.csv",
        mime="text/csv"
    )
