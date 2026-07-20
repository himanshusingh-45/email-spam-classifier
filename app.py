import joblib
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Email Spam Classifier", page_icon="📧", layout="centered"
)


# Load Trained Assets
@st.cache_resource
def load_assets():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_assets()
except Exception as e:
    st.error(
        "Model or Vectorizer file not found! Please run `python train_model.py` first."
    )
    st.stop()

# Header Section
st.title("📧 Email & SMS Spam Detector")
st.write(
    "Paste any email or message below to check whether it's **Spam** or **Ham (Legitimate)**."
)

# User Input
email_text = st.text_area(
    "Email / Message Content:",
    height=180,
    placeholder="e.g., Congratulations! You have won a $1,000 Walmart gift card. Click here to claim...",
)

col1, col2 = st.columns([1, 4])

with col1:
    predict_btn = st.button("Analyze Message", type="primary")

if predict_btn:
    if not email_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess & Predict
        text_vectorized = vectorizer.transform([email_text])
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]

        spam_prob = probabilities[1] * 100
        ham_prob = probabilities[0] * 100

        st.divider()

        # Output Results
        if prediction == 1:
            st.error("🚨 **Result: SPAM DETECTED**")
            st.metric(label="Spam Confidence", value=f"{spam_prob:.1f}%")
        else:
            st.success("✅ **Result: HAM (Legitimate Email)**")
            st.metric(label="Legitimate Confidence", value=f"{ham_prob:.1f}%")

        # Visual breakdown
        st.caption("Probability Distribution")
        st.progress(int(spam_prob))

# Sidebar Info for Resume / Showcase
st.sidebar.header("About This Project")
st.sidebar.info(
    """
- **Algorithm:** Multinomial Naive Bayes
- **Feature Extraction:** TF-IDF Vectorizer
- **Dataset:** SMS / Email Spam Collection
- **Accuracy:** ~98.4%
"""
)
st.sidebar.caption("Built with Python & Streamlit")