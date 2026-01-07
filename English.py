import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="English Show App",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("Anglais 3.txt")

st.title("Data Displayment")
st.write("Here is the dataset")
st.write(df)

# --------------------------------------------------
# TARGET
# --------------------------------------------------
df["Study_habits_affect_performance"] = df[
    "Study_habits_affect_performance"
].map({"Yes": 1, "No": 0})

# --------------------------------------------------
# DROP ID
# --------------------------------------------------
st.write("Drop ID")
df = df.drop(columns=["Student"])
st.write(df)

# --------------------------------------------------
# ONE-HOT ENCODING (kept for display only)
# --------------------------------------------------
st.write("One-hot encoding")
df = pd.get_dummies(
    df,
    columns=["Gender", "Phone_use_while_studying", "Study_rhythm"],
    drop_first=True
)
st.write(df)

# --------------------------------------------------
# FEATURES (✅ ADD NEGATIVE VARIABLE HERE)
# --------------------------------------------------
features = [
    "Study_hours_per_week",
    "Sleep_hours_per_night",
    "Social_media_scroll_hours_per_day"
]

X_train = df[features]
y = df["Study_habits_affect_performance"]

# --------------------------------------------------
# MODEL
# --------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train, y)

# --------------------------------------------------
# STREAMLIT APP
# --------------------------------------------------
st.title("What features affect the most your academic performance?")

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.slider("Study hours per week", 0, 20, 7)

with col2:
    sleep_hours = st.slider("Sleep hours per night", 3, 9, 7)

with col3:
    social_media_hours = st.slider(
        "Social media scrolling (hours/day)",
        0.0, 6.0, 2.0, 0.1
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
input_data = np.array([[study_hours, sleep_hours, social_media_hours]])
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0][1]

if prediction == 1:
    st.success(f"✅ Good academic performance likely (probability = {proba:.2f})")
    st.balloons()
else:
    st.error(f"⚠️ Risk of poor academic performance (probability = {proba:.2f})")


st.subheader("🎯 Risk Indicator")

risk = 1 - proba

st.progress(int(proba * 100))
st.caption(f"Predicted success probability: {proba*100:.1f}%")

# --------------------------------------------------
# INTERPRETATION
# --------------------------------------------------
st.subheader("📈 Model Interpretation")

coeffs = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
})

st.write(coeffs)

st.info(
    """
    **Interpretation:**
    - Positive coefficient → increases probability of good performance
    - Negative coefficient → decreases probability
    """
)


import matplotlib.pyplot as plt

st.subheader("📊 Feature Influence")

fig, ax = plt.subplots()
ax.bar(coeffs["Feature"], coeffs["Coefficient"])
ax.axhline(0)
ax.set_ylabel("Coefficient value")
ax.set_title("Impact of Variables on Academic Performance")
plt.xticks(rotation=20)

st.pyplot(fig)


st.subheader("🔍 What-if Analysis")

fixed_study = study_hours
fixed_sleep = sleep_hours

social_range = np.linspace(0, 6, 50)
X_sim = np.column_stack([
    np.full(50, fixed_study),
    np.full(50, fixed_sleep),
    social_range
])

X_sim_scaled = scaler.transform(X_sim)
probas = model.predict_proba(X_sim_scaled)[:,1]

fig, ax = plt.subplots()
ax.plot(social_range, probas)
ax.set_xlabel("Social media hours per day")
ax.set_ylabel("Probability of good performance")
ax.set_title("Effect of Social Media Usage")

st.pyplot(fig)

st.divider()
st.header("📌 Conclusion")

st.divider()
st.markdown(
    "<h3 style='text-align:center;'>🙏 Thank you for your attention</h3>",
    unsafe_allow_html=True
)


