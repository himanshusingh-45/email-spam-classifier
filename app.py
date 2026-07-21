import io
import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="MailSentinel AI | Enterprise Spam Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Sleek Glassmorphism & UI Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    .metric-card {
        background: #1e222d;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2e364f;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load Trained ML Assets
@st.cache_resource
def load_assets():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_assets()
except Exception as e:
    st.error(
        "❌ Trained assets missing! Run `python train_model.py` in your terminal first."
    )
    st.stop()

# Header Banner
st.title("🛡️ MailSentinel AI")
st.caption(
    "Enterprise-grade NLP Spam Detection Engine | Powered by Scikit-learn & TF-IDF"
)
st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(
    ["🔍 Live Message Scanner", "📁 Bulk Batch Processing", "📊 Model Insights"]
)

# ================= TAB 1: SINGLE MESSAGE SCANNER =================
with tab1:
    st.subheader("Single Message Analyzer")

    email_text = st.text_area(
        "Enter Mail Body or SMS Content:",
        height=160,
        placeholder="Paste message content here to analyze threat probability...",
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    if st.button("⚡ Scan Message Threat"):
        if not email_text.strip():
            st.warning("⚠️ Please provide text input before running analysis.")
        else:
            text_vec = vectorizer.transform([email_text])
            prediction = model.predict(text_vec)[0]
            probs = model.predict_proba(text_vec)[0]

            spam_score = probs[1] * 100
            ham_score = probs[0] * 100

            word_count = len(email_text.split())
            char_count = len(email_text)

            st.markdown("### 📊 Scan Results")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Detection Outcome", "SPAM" if prediction == 1 else "HAM (Safe)")
            with m2:
                st.metric("Spam Risk Score", f"{spam_score:.1f}%")
            with m3:
                st.metric("Total Words", word_count)
            with m4:
                st.metric("Total Characters", char_count)

            if prediction == 1:
                st.error(
                    f"🚨 **High Threat Detected!** Confidence Level: **{spam_score:.2f}%**"
                )
            else:
                st.success(
                    f"✅ **Legitimate Message.** Safety Score: **{ham_score:.2f}%**"
                )

            st.write("**Threat Risk Level:**")
            st.progress(int(spam_score))

# ================= TAB 2: BULK BATCH PROCESSING =================
with tab2:
    st.subheader("Batch Spam Detector (CSV Analysis)")
    st.write(
        "Upload a `.csv` file containing emails/messages to process bulk predictions."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV (must contain a column named `message` or `text`)",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            col_name = "message" if "message" in df.columns else ("text" if "text" in df.columns else None)

            if col_name is None:
                st.error("CSV file must have a column named either `message` or `text`!")
            else:
                st.info(f"Loaded {len(df)} records. Processing bulk analysis...")

                vec_data = vectorizer.transform(df[col_name].astype(str))
                df["Prediction"] = model.predict(vec_data)
                df["Prediction"] = df["Prediction"].map({0: "Legitimate (Ham)", 1: "Spam"})
                df["Spam Probability (%)"] = (model.predict_proba(vec_data)[:, 1] * 100).round(2)

                st.write("### 📋 Prediction Results")
                st.dataframe(df[[col_name, "Prediction", "Spam Probability (%)"]], use_container_width=True)

                # Download Button
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Classified Results CSV",
                    data=csv_buffer.getvalue(),
                    file_name="spam_predictions_report.csv",
                    mime="text/csv",
                )
        except Exception as err:
            st.error(f"Error processing file: {err}")

# ================= TAB 3: MODEL INSIGHTS & METRICS =================
with tab3:
    st.subheader("Machine Learning Performance Overview")
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("#### 🎯 Performance Benchmarks")
        st.json(
            {
                "Model Architecture": "Multinomial Naive Bayes",
                "Feature Extraction": "TF-IDF Vectorizer (max_features=3000)",
                "Overall Accuracy": "98.57%",
                "Precision (Spam)": "0.99",
                "Recall (Spam)": "0.90",
                "F1-Score (Spam)": "0.94",
            }
        )

    with col_b:
        st.write("#### 🛠 System Design & Pipeline")
        st.markdown(
            """
            1. **Preprocessing:** Lowercasing, stop-word filtering & noise reduction.
            2. **Vectorization:** TF-IDF calculation for high-dim textual representation.
            3. **Inference:** Probabilistic classification via Bayes Theorem.
            4. **Output:** Real-time risk probability calculation.
            """
        )

# Sidebar Info
st.sidebar.title("📌 Developer Panel")
st.sidebar.info(
    """
**MailSentinel AI**  
Developed by **Himanshu Singh**  
*Integrated ML Infrastructure for Resume Portfolio*
"""
)
st.sidebar.markdown("---")
st.sidebar.caption("v2.1.0 Enterprise Edition")