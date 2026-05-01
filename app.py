"""
Streamlit Web App for Email Spam Classification
================================================
Run with: streamlit run app.py
"""

import streamlit as st
from src.predict import SpamPredictor

# Page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Title and description
st.title("📧 Email Spam Classifier")
st.markdown("""
This app classifies emails as **SPAM** or **HAM** (not spam) using a machine learning model.
""")

# Initialize predictor (cache for performance)
@st.cache_resource
def load_predictor():
    return SpamPredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input section
st.subheader("✉️ Enter Email Text")
email_text = st.text_area(
    "Paste your email content here:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize..."
)

# Prediction button
if st.button("🔍 Classify Email", type="primary"):
    if email_text.strip():
        with st.spinner("Classifying..."):
            try:
                # Get prediction
                result = predictor.predict(email_text)
                
                # Get probability
                proba = predictor.predict_proba(email_text)
                
                # Display results
                st.divider()
                st.subheader("📊 Results")
                
                # Result with styling
                if result == "SPAM":
                    st.error(f"### 🚫 Result: **{result}**")
                else:
                    st.success(f"### ✅ Result: **{result}**")
                
                # Show probabilities
                st.write("**Confidence Scores:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Ham (Not Spam)", f"{proba['HAM']:.1%}")
                with col2:
                    st.metric("Spam", f"{proba['SPAM']:.1%}")
                
                # Progress bar visualization (st.progress expects 0.0-1.0)
                st.write("**Probability Distribution:**")
                st.progress(proba['HAM'], text=f"Ham: {proba['HAM']:.1%}")
                st.progress(proba['SPAM'], text=f"Spam: {proba['SPAM']:.1%}")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.warning("Please enter some email text to classify.")

# Sample emails section
st.divider()
st.subheader("💡 Try Sample Emails")

sample_emails = {
    "Obvious Spam": "Congratulations! You have won a $1,000,000 prize! Click here to claim your reward now!",
    "Promotional Spam": "Buy now! Limited time offer! 50% off all items. Act fast!",
    "Phishing Attempt": "URGENT: Your account has been compromised. Verify your identity immediately!",
    "Legitimate Email": "Hey, are we still meeting tomorrow for lunch? Let me know if the time works.",
    "Work Email": "Thanks for the project update. Please send me the report by Friday.",
    "Meeting Request": "Meeting scheduled for 3pm. Please confirm your attendance."
}

for label, text in sample_emails.items():
    if st.button(label, key=f"btn_{label}"):
        # Show the sample and classify
        st.info(f"**{label}:** {text[:100]}...")
        result = predictor.predict(text)
        proba = predictor.predict_proba(text)
        
        if result == "SPAM":
            st.error(f"**Prediction: SPAM** (Confidence: {proba['SPAM']:.1%})")
        else:
            st.success(f"**Prediction: HAM** (Confidence: {proba['HAM']:.1%})")

# Footer
st.divider()
st.caption("""
---
**Model Information:**
- Algorithm: Support Vector Machine (SVM) with TF-IDF features
- Accuracy: ~99%
- Features: Cross-validation, hyperparameter tuning, class balancing
""")