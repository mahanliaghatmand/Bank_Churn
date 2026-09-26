import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

ROOT = Path(__file__).parent
NUM_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CAT_COLS = ["Geography", "Gender"]

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")
st.markdown(
    """<style>
    .stApp{background:linear-gradient(180deg,#0f1116 0%,#131722 100%)}
    [data-testid="stMetric"]{background:#1b1f2b;border:1px solid #2a3040;
    padding:14px 18px;border-radius:14px}
    [data-testid="stMetricLabel"]{color:#9aa4b2}
    h1,h2,h3{color:#e8ecf3}
    </style>""",
    unsafe_allow_html=True,
)


# ---------- Loading ----------
@st.cache_resource
def load_artifacts():
    # Same architecture as the training notebook
    model = nn.Sequential(
        nn.Linear(13, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Dropout(0.8), nn.Linear(32, 1), nn.Sigmoid(),
    )
    model.load_state_dict(torch.load(ROOT / "artifacts" / "model_bank_costumers.pt", map_location="cpu"))
    model.eval()
    with open(ROOT / "artifacts" / "OneHotEncoder.pkl", "rb") as f:
        ohe = pickle.load(f)
    with open(ROOT / "artifacts" / "StandardScaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, ohe, scaler


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "dataset" / "Customer-Churn-Records.csv")


model, ohe, scaler = load_artifacts()
data = load_data()


def predict(df: pd.DataFrame) -> np.ndarray:
    """Returns churn probability for each row (scaler + one-hot encoder, same order as training)."""
    x = np.hstack([df[NUM_COLS].to_numpy(float), ohe.transform(df[CAT_COLS])])
    x = scaler.transform(x)
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).squeeze(1).numpy()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("🧾 Customer profile")
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    age = st.slider("Age", 18, 92, 38)
    credit_score = st.slider("Credit score", 350, 850, 650)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 260000.0, 75000.0, step=1000.0)
    salary = st.number_input("Estimated salary", 0.0, 200000.0, 100000.0, step=1000.0)
    num_products = st.slider("Number of products", 1, 4, 2)
    has_card = st.toggle("Has credit card", value=True)
    is_active = st.toggle("Active member", value=True)
    st.divider()
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Probability above which the customer is flagged as likely to churn.")

customer = pd.DataFrame([{
    "CreditScore": credit_score, "Geography": geography, "Gender": gender, "Age": age, "Tenure": tenure,
    "Balance": balance, "NumOfProducts": num_products, "HasCrCard": int(has_card),
    "IsActiveMember": int(is_active), "EstimatedSalary": salary,
}])

# ---------- Header ----------
st.title("🏦 Bank Churn Predictor")
st.caption("A PyTorch neural network trained on 10,000 bank customer records.")

tab_pred, tab_batch, tab_data, tab_about = st.tabs(["🔮 Prediction", "📁 Batch", "📊 Data insights", "ℹ️ About"])

# ---------- Single prediction ----------
with tab_pred:
    prob = float(predict(customer)[0])
    positive = prob >= threshold
    accent = "#ff5c7a" if positive else "#3ddc97"

    top1, top2, top3 = st.columns(3)
    top1.metric("Churn probability", f"{prob:.1%}")
    top2.metric("Threshold", f"{threshold:.0%}")
    top3.metric("Verdict", "At risk" if positive else "Likely to stay")

    left, right = st.columns([1, 1.2])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Risk of churning"},
            gauge={
                "axis": {"range": [0, 100]}, "bar": {"color": accent, "thickness": 0.35},
                "bgcolor": "#1b1f2b",
                "steps": [{"range": [0, threshold * 100], "color": "rgba(61,220,151,.15)"},
                          {"range": [threshold * 100, 100], "color": "rgba(255,92,122,.15)"}],
                "threshold": {"line": {"color": "#e8ecf3", "width": 3}, "value": threshold * 100},
            },
        ))
        fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#e8ecf3")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Compared to the customer base")
        avg = data[NUM_COLS].mean()
        bars = px.bar(
            x=["Age", "Credit score", "Balance", "Salary"],
            y=[age - avg["Age"], credit_score - avg["CreditScore"],
               balance - avg["Balance"], salary - avg["EstimatedSalary"]],
            labels={"x": "", "y": "Difference from average"},
            color=[age - avg["Age"], credit_score - avg["CreditScore"],
                   balance - avg["Balance"], salary - avg["EstimatedSalary"]],
            color_continuous_scale="RdYlGn",
        )
        bars.update_layout(showlegend=False, coloraxis_showscale=False, height=300,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8ecf3")
        st.plotly_chart(bars, use_container_width=True)

    if positive:
        st.error("**At risk of churning** — this customer profile matches patterns of customers who left.")
    else:
        st.success("**Likely to stay** — this customer profile matches patterns of retained customers.")
    st.warning("Educational demo only — not a substitute for real retention analysis.", icon="⚠️")

# ---------- Batch prediction ----------
with tab_batch:
    st.write(f"Upload a CSV with these columns: `{', '.join(CAT_COLS + NUM_COLS)}`")
    file = st.file_uploader("CSV file", type="csv")
    if file:
        try:
            batch = pd.read_csv(file)
            batch["churn_probability"] = predict(batch).round(4)
            batch["prediction"] = np.where(batch["churn_probability"] >= threshold, "Will churn", "Will stay")
            st.dataframe(batch, use_container_width=True)
            st.download_button("⬇️ Download results", batch.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process the file: {e}")

# ---------- Data insights ----------
with tab_data:
    m1, m2, m3 = st.columns(3)
    m1.metric("Customers", f"{len(data):,}")
    m2.metric("Churn rate", f"{data['Exited'].mean():.1%}")
    m3.metric("Features", len(NUM_COLS + CAT_COLS))

    col_a, col_b = st.columns(2)
    with col_a:
        by_geo = data.groupby("Geography")["Exited"].mean().reset_index()
        fig1 = px.bar(by_geo, x="Geography", y="Exited", title="Churn rate by geography",
                      color="Exited", color_continuous_scale="Reds")
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8ecf3")
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        corr = data[NUM_COLS + ["Exited"]].corr()
        fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e8ecf3")
        st.plotly_chart(fig2, use_container_width=True)

    feature = st.selectbox("Feature distribution", ["Age", "CreditScore", "Balance", "EstimatedSalary"])
    hist = px.histogram(data.assign(Exited=data["Exited"].map({0: "Stayed", 1: "Churned"})), x=feature,
                        color="Exited", barmode="overlay", opacity=0.65, nbins=50,
                        color_discrete_map={"Stayed": "#3ddc97", "Churned": "#ff5c7a"})
    hist.add_vline(x=customer[feature].iloc[0], line_dash="dash", annotation_text="Customer")
    hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8ecf3")
    st.plotly_chart(hist, use_container_width=True)

# ---------- About ----------
with tab_about:
    st.markdown(
        """
**Model:** fully-connected network `13 → 256 → 128 → 64 → 32 → 1` (ReLU, Dropout 0.8, Sigmoid), trained with BCE loss and Adam.

**Preprocessing:** one-hot encoding for geography and gender, standard scaling on all features, SMOTE to balance classes during training.

**Limitations**
- Training data was balanced with SMOTE, so probabilities are not calibrated to the real churn rate (about 20% in this dataset).
- Dropout is set very high (0.8), which can make single-prediction outputs a bit noisy.
- Not a substitute for a full retention/CLV analysis.
        """
    )
