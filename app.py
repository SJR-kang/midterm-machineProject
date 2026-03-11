import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(page_title="Tweet Moderation", page_icon="🔍")

# Simple title
st.title("🔍 Tweet Moderation System")
st.markdown("Analyze tweets for harmful content")

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        # Try loading from different possible paths
        possible_paths = [
            'models/vectorizer.pkl',
            'vectorizer.pkl',
            '../models/vectorizer.pkl'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                vec_path = path.replace('vectorizer.pkl', '')
                vectorizer = joblib.load(path)
                scaler = joblib.load(vec_path + 'scaler.pkl')
                model = joblib.load(vec_path + 'random_forest_model.pkl')
                policy = joblib.load(vec_path + 'recommendation_policy.pkl')
                return vectorizer, scaler, model, policy
        
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Load data
@st.cache_data
def load_data():
    try:
        # Try loading from different paths
        possible_paths = ['Tweets_reclassified.csv', 'data/Tweets_reclassified.csv']
        for path in possible_paths:
            if os.path.exists(path):
                return pd.read_csv(path)
        return None
    except:
        return None

# Load everything
vectorizer, scaler, model, policy = load_models()
df = load_data()

# Simple sidebar
with st.sidebar:
    st.header("ℹ️ Info")
    if model is not None:
        st.success("✅ Model ready")
    else:
        st.warning("⚠️ Using fallback mode")
    
    if df is not None:
        st.info(f"📊 Dataset: {len(df)} tweets")

# Main input
tweet = st.text_area("Enter a tweet:", height=100)

# Recommendation policy (define it here)
recommendation_policy = {
    0: {"action": "Allow content", "priority": "Low", "reason": "No abusive language"},
    1: {"action": "Allow content", "priority": "Low", "reason": "No abusive language"},
    2: {"action": "Flag for review", "priority": "Medium", "reason": "Offensive language"},
    3: {"action": "Hide and warn", "priority": "High", "reason": "Hate speech"},
    4: {"action": "Remove and alert", "priority": "Critical", "reason": "Threat detected"},
}

if st.button("Analyze", type="primary"):
    if tweet:
        with st.spinner("Analyzing..."):
            try:
                if model is not None:
                    # Use actual model
                    tweet_vec = vectorizer.transform([tweet])
                    tweet_scaled = scaler.transform(tweet_vec)
                    pred = model.predict(tweet_scaled)[0]
                else:
                    # Simple fallback
                    tweet_lower = tweet.lower()
                    threat_words = ['kill', 'die', 'murder', 'shoot', 'bomb']
                    hate_words = ['nigger', 'wetback', 'spic', 'chink', 'kike']
                    offense_words = ['fuck', 'shit', 'bitch', 'cunt', 'asshole']
                    
                    if any(word in tweet_lower for word in threat_words):
                        pred = 4
                    elif any(word in tweet_lower for word in hate_words):
                        pred = 3
                    elif any(word in tweet_lower for word in offense_words):
                        pred = 2
                    else:
                        pred = 0
                
                # Show result
                result = recommendation_policy.get(pred, recommendation_policy[0])
                
                # Color based on class
                colors = {0: "green", 1: "green", 2: "orange", 3: "red", 4: "darkred"}
                icons = {0: "✅", 1: "✅", 2: "⚠️", 3: "🚫", 4: "🔴"}
                
                st.markdown(f"""
                <div style="padding:20px; border-radius:10px; background-color:{colors[pred]}20; border-left:5px solid {colors[pred]}">
                    <h3>{icons[pred]} Class {pred}</h3>
                    <p><strong>Action:</strong> {result['action']}</p>
                    <p><strong>Priority:</strong> {result['priority']}</p>
                    <p><strong>Reason:</strong> {result['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a tweet")

# Show sample data if available
if df is not None:
    with st.expander("📊 View Dataset Sample"):
        st.dataframe(df.head(10))
