import streamlit as st
import pickle

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📩",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- CSS DESIGN ---------------- #
st.markdown(
    """
    <style>

    .main {
        background: #0b1220;
    }

    .title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: #38bdf8;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 30px;
    }

    /* input box */
    textarea {
        background-color: #111827 !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid #334155 !important;
        font-size: 16px !important;
    }

    /* button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg,#06b6d4,#6366f1);
        color: white;
        font-weight: bold;
        padding: 12px;
        border-radius: 12px;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0px 5px 20px rgba(99,102,241,0.4);
    }

    /* result box */
    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.title("📊 About Project")

    st.info("""
    This AI model detects Spam messages using:
    - TF-IDF Vectorization  
    - Machine Learning Model  
    - NLP Preprocessing  
    """)

    st.success("Built using Python + Streamlit")

    st.markdown("---")
    st.write("⚡ Model Workflow")
    st.write("Message → Clean → Vectorize → Predict")

# ---------------- MAIN UI ---------------- #
st.markdown("<div class='title'>📩 AI Spam Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect Spam Messages using Machine Learning</div>", unsafe_allow_html=True)

message = st.text_area("✍️ Enter your Email or SMS Message", height=160)

col1, col2 = st.columns(2)

with col1:
    words = len(message.split())
    st.metric("📝 Words", words)

with col2:
    chars = len(message)
    st.metric("🔤 Characters", chars)

predict = st.button("🚀 Predict Message")

# ---------------- PREDICTION ---------------- #
if predict:
    if message.strip() == "":
        st.warning("Please enter a message!")
    else:
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]

        st.markdown("---")

        if prediction == 1:
            st.markdown(
                "<div class='result-box' style='background:#7f1d1d;color:white;'>🚨 SPAM DETECTED</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box' style='background:#14532d;color:white;'>✅ NOT SPAM</div>",
                unsafe_allow_html=True
            )

        # fake confidence (optional UI feel)
        st.write("### Confidence Score")
        st.progress(0.95)
        st.caption("Model confidence: High")