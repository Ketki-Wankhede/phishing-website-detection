
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🎣",
    layout="wide"
)

@st.cache_resource
def load_bundle():
    return joblib.load("random_forest_model.pkl")

st.title("🎣 Phishing Website Detector")
st.write(
    "This machine learning demo estimates whether a website "
    "may be phishing or legitimate."
)

try:
    bundle = load_bundle()
    model = bundle["model"]
    all_columns = bundle["feature_names"]
    top_features = bundle["top_features"]
    medians = bundle["medians"]
    ranges = bundle["ranges"]
except Exception as e:
    st.error("Could not load the trained model.")
    st.exception(e)
    st.stop()

st.subheader("Website Features")
st.info(
    "Adjust the eight important features. Other features "
    "will use typical values from the training dataset."
)

user_input = {}
cols = st.columns(2)

for i, feat in enumerate(top_features):
    lo, hi = ranges[feat]
    default = float(medians[feat])

    with cols[i % 2]:
        if lo == hi:
            user_input[feat] = default
        elif (lo in (0, -1)) and hi == 1:
            user_input[feat] = st.slider(
                feat,
                min_value=float(lo),
                max_value=float(hi),
                value=max(float(lo), min(float(hi), default)),
                step=0.01
            )
        else:
            user_input[feat] = st.slider(
                feat,
                min_value=float(lo),
                max_value=float(hi),
                value=max(float(lo), min(float(hi), default)),
            )

if st.button("Check Website", type="primary"):
    row = medians.copy()
    row.update(user_input)
    input_df = pd.DataFrame([row])[all_columns]

    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.divider()
        if pred == 1:
            st.error(
                f"🚨 Likely Phishing — model score: {prob * 100:.1f}%"
            )
        else:
            st.success(
                f"✅ Likely Legitimate — model score: "
                f"{(1 - prob) * 100:.1f}%"
            )

        st.progress(float(prob))
        st.caption("The bar shows the model's estimated phishing probability.")
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

st.divider()
st.caption(
    "Model: Random Forest. Test accuracy: 98.50%. "
    "This is a demonstration, not a guarantee of website safety."
)
