import streamlit as st
import pickle

# Load model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# App UI
st.title("📩 Spam Message Classifier")
st.write("Enter a message and check if it's Spam or Not Spam")

# Input box
message = st.text_area("Enter your message here")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message!")
    else:
        # Transform text
        data = vectorizer.transform([message])

        # Predict
        prediction = model.predict(data)[0]

        # Output
        if prediction == 1:
            st.error("🚨 This is SPAM message")
        else:
            st.success("✅ This is NOT spam message")