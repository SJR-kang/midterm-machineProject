import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os

st.set_page_config(page_title="Tweet Moderation System", page_icon="🔍", layout="wide")

# Initialize session state
if "tweet_input" not in st.session_state:
    st.session_state.tweet_input = ""
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

st.title("🔍 AI-Powered Tweet Moderation System")
st.markdown("Enter a tweet below to get instant analysis and suggested actions.")

# Class names and policy (as before)
CLASS_NAMES = {0: "Religious hate speech", 1: "Non-abusive / General",
               2: "Racist content", 3: "Discrimination / Threats", 4: "NSFW / Explicit content"}
RECOMMENDATION_POLICY = {
    0: {"action": "Flag for review", "priority": "High", "reason": "Religious hate speech detected"},
    1: {"action": "Allow content", "priority": "Low", "reason": "Non-abusive / General content"},
    2: {"action": "Hide and warn user", "priority": "High", "reason": "Racist content detected"},
    3: {"action": "Remove and alert moderators", "priority": "Critical", "reason": "Threatening or violent content"},
    4: {"action": "Remove content", "priority": "Critical", "reason": "NSFW / Explicit content detected"},
}

# Load models (unchanged)
@st.cache_resource
def load_models():
    try:
        if not os.path.exists('models'):
            st.sidebar.error("❌ 'models' folder not found!")
            return None, None, None
        vectorizer = joblib.load('models/vectorizer.pkl')
        model = joblib.load('models/best_model.pkl')
        st.sidebar.success("✅ Models loaded successfully!")
        return vectorizer, model, RECOMMENDATION_POLICY
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {str(e)}")
        return None, None, None

vectorizer, model, policy = load_models()

# Sidebar (unchanged, but uses CLASS_NAMES)
with st.sidebar:
    st.header("📊 About the System")
    st.info("Classification categories and policy...")  # (shortened for brevity)
    # ... (your existing sidebar content)

# Main content
st.header("📝 Tweet Analysis")
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    # Text area – now using key="tweet_input" (no separate value parameter needed)
    st.text_area("Enter your tweet to analyze:", height=150,
                 placeholder="Type or paste a tweet here...", key="tweet_input")

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        analyze_button = st.button("🔍 Analyze Tweet", type="primary", use_container_width=True)

    # Cleaning function
    def clean_tweet(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Analysis function (identical to before, but uses st.session_state.tweet_input)
    def analyze_tweet(tweet_text):
        if not tweet_text:
            st.warning("Please enter a tweet.")
            return
        with st.spinner("Analyzing..."):
            try:
                clean = clean_tweet(tweet_text)
                if not clean:
                    st.warning("Tweet empty after cleaning.")
                    return
                if model and vectorizer:
                    vec = vectorizer.transform([clean])
                    pred_class = model.predict(vec)[0]
                    confidence = max(model.predict_proba(vec)[0]) if hasattr(model, "predict_proba") else 0.95
                else:
                    # fallback (simplified)
                    threat_words = ['kill', 'die', 'murder', 'shoot', 'bomb']
                    hate_words = ['nigger', 'wetback', 'spic', 'chink', 'kike', 'raghead', 'sand nigger']
                    lower = tweet_text.lower()
                    if any(w in lower for w in threat_words):
                        pred_class = 3
                    elif any(w in lower for w in hate_words):
                        pred_class = 2
                    else:
                        pred_class = 1
                    confidence = 0.85

                res = policy.get(pred_class, policy[1])
                desc = CLASS_NAMES.get(pred_class, "Unknown")

                # Color-coded result box (dark text for readability)
                color_map = {1: ("#d4edda", "#c3e6cb", "✅"),
                             0: ("#fff3cd", "#ffeeba", "⚠️"),
                             2: ("#fff3cd", "#ffeeba", "⚠️"),
                             3: ("#f8d7da", "#f5c6cb", "🚫"),
                             4: ("#f8d7da", "#f5c6cb", "🔴")}
                bg_color, border_color, icon = color_map.get(pred_class, ("#f8d7da", "#f5c6cb", "🔴"))

                st.markdown("---")
                st.subheader("📊 Analysis Results")
                st.markdown(f"""
                <div style="padding:25px; border-radius:10px; background-color:{bg_color}; border:2px solid {border_color}; margin:10px 0;">
                    <h2 style="color:#212529;">{icon} Class {pred_class}: {desc}</h2>
                    <p style="font-size:18px; color:#212529;"><strong>Action:</strong> {res['action']}</p>
                    <p style="font-size:18px; color:#212529;"><strong>Priority:</strong> {res['priority']}</p>
                    <p style="font-size:18px; color:#212529;"><strong>Reason:</strong> {res['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {confidence:.1%}")
                st.progress(float(confidence))
                with st.expander("📝 Cleaned tweet"):
                    st.write(clean)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Manual analyze button
    if analyze_button:
        analyze_tweet(st.session_state.tweet_input)
        st.session_state.run_analysis = False

    # Trigger analysis if requested by a sample button
    if st.session_state.run_analysis and st.session_state.tweet_input:
        analyze_tweet(st.session_state.tweet_input)
        st.session_state.run_analysis = False

    # Sample tweets – now updating st.session_state.tweet_input directly
    st.markdown("---")
    st.subheader("🧪 Try Sample Tweets")
    sample_tweets = {
        "Clean": "I love this beautiful day! 😊",
        "Religious Hate": "Muslims are terrorists and should be banned",
        "Racist": "Go back to your country you nigger",
        "Threat": "I'm going to kill you",
        "NSFW": "Check out my naked pics at link in bio"
    }

    cols = st.columns(len(sample_tweets))
    for idx, (category, tweet) in enumerate(sample_tweets.items()):
        with cols[idx]:
            if st.button(f"📋 {category}", key=f"sample_{idx}", use_container_width=True):
                st.session_state.tweet_input = tweet   # updates the text area (because key="tweet_input")
                st.session_state.run_analysis = True
                st.rerun()

# Footer (unchanged)
st.markdown("---")
st.header("📋 Moderation Policy Guidelines")
policy_df = pd.DataFrame([
    {"Class": "0 – Religious hate speech", "Action": "Flag for review", "Priority": "High", "Description": "Religious hate speech detected"},
    {"Class": "1 – Non-abusive / General", "Action": "Allow content", "Priority": "Low", "Description": "Non-abusive content"},
    {"Class": "2 – Racist content", "Action": "Hide and warn user", "Priority": "High", "Description": "Racist content detected"},
    {"Class": "3 – Discrimination / Threats", "Action": "Remove and alert moderators", "Priority": "Critical", "Description": "Threatening or violent content"},
    {"Class": "4 – NSFW / Explicit", "Action": "Remove content", "Priority": "Critical", "Description": "NSFW / Explicit content detected"},
])
st.table(policy_df)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; padding: 10px;'>Tweet Moderation System v2.3 | Powered by Machine Learning</div>", unsafe_allow_html=True)
