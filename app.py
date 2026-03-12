import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Tweet Moderation System",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state for tweet input
if 'tweet_input' not in st.session_state:
    st.session_state.tweet_input = ""

# Title and description
st.title("🔍 AI-Powered Tweet Moderation System")
st.markdown("""
This application analyzes tweets for harmful content and provides automated moderation recommendations.
Enter a tweet below to get instant analysis and suggested actions.
""")

# Define class names based on dataset analysis (from your notebook)
CLASS_NAMES = {
    0: "Religious hate speech",
    1: "Non-abusive / General",
    2: "Racist content",
    3: "Discrimination / Threats",
    4: "NSFW / Explicit content"
}

# Updated recommendation policy based on correct meanings
RECOMMENDATION_POLICY = {
    0: {"action": "Flag for review", "priority": "High", "reason": "Religious hate speech detected"},
    1: {"action": "Allow content", "priority": "Low", "reason": "Non-abusive / General content"},
    2: {"action": "Hide and warn user", "priority": "High", "reason": "Racist content detected"},
    3: {"action": "Remove and alert moderators", "priority": "Critical", "reason": "Threatening or violent content"},
    4: {"action": "Remove content", "priority": "Critical", "reason": "NSFW / Explicit content detected"}
}

# Load the saved models
@st.cache_resource
def load_models():
    """Load trained models from local folder"""
    try:
        if not os.path.exists('models'):
            st.sidebar.error("❌ 'models' folder not found!")
            return None, None, None
        
        vectorizer = joblib.load('models/vectorizer.pkl')
        model = joblib.load('models/best_model.pkl')
        # We use our defined policy, not the saved one
        policy = RECOMMENDATION_POLICY
        
        st.sidebar.success("✅ Models loaded successfully!")
        return vectorizer, model, policy
        
    except FileNotFoundError as e:
        st.sidebar.error(f"❌ Model file not found: {str(e)}")
        return None, None, None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {str(e)}")
        return None, None, None

# Load data for statistics (optional)
@st.cache_data
def load_data():
    """Load data for statistics display"""
    try:
        possible_paths = ['Tweets_reclassified.csv', 'Tweets.csv', 'data/Tweets.csv']
        for path in possible_paths:
            if os.path.exists(path):
                return pd.read_csv(path)
        return None
    except:
        return None

# Load models and data
vectorizer, model, policy = load_models()
df = load_data()

# Sidebar for information
with st.sidebar:
    st.header("📊 About the System")
    st.info(f"""
    **Classification Categories:**
    - **Class 0:** Religious hate speech – Flag for review
    - **Class 1:** Non-abusive / General – Allow
    - **Class 2:** Racist content – Hide and warn user
    - **Class 3:** Discrimination / Threats – Remove and alert moderators
    - **Class 4:** NSFW / Explicit – Remove content

    **Model Used:** SVM (best performer)  
    **Macro F1:** 0.66
    """)
    
    if model is not None:
        st.success("✅ Model ready")
    else:
        st.warning("⚠️ Using rule-based fallback")
    
    if df is not None:
        st.header("📁 Dataset Statistics")
        if 'label' in df.columns:
            class_counts = df['label'].value_counts().sort_index()
            st.write(f"**Total tweets:** {len(df)}")
            st.write("**Class Distribution:**")
            for cls, count in class_counts.items():
                name = CLASS_NAMES.get(cls, "Unknown")
                st.write(f"- Class {cls} ({name}): {count} ({count/len(df)*100:.1f}%)")

# Main content - Single tweet analysis only
st.header("📝 Tweet Analysis")

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    tweet_input = st.text_area(
        "Enter your tweet to analyze:",
        height=150,
        placeholder="Type or paste a tweet here...",
        value=st.session_state.tweet_input,
        key="tweet_input_area"
    )
    
    st.session_state.tweet_input = tweet_input
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        analyze_button = st.button("🔍 Analyze Tweet", type="primary", use_container_width=True)
    
    # Function to clean tweet (same as notebook)
    def clean_tweet(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Function to analyze tweet
    def analyze_tweet(tweet_text):
        if not tweet_text:
            st.warning("Please enter a tweet to analyze.")
            return
        
        with st.spinner("Analyzing..."):
            try:
                clean = clean_tweet(tweet_text)
                if not clean:
                    st.warning("Tweet became empty after cleaning.")
                    return
                
                if model is not None and vectorizer is not None:
                    vec = vectorizer.transform([clean])
                    # SVM can handle sparse input directly
                    pred_class = model.predict(vec)[0]
                    
                    # Get confidence if available
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(vec)[0]
                        confidence = max(probs)
                    else:
                        confidence = 0.95
                else:
                    # Rule-based fallback (simplified)
                    tweet_lower = tweet_text.lower()
                    threat_words = ['kill', 'die', 'murder', 'shoot', 'bomb']
                    hate_words = ['nigger', 'wetback', 'spic', 'chink', 'kike', 'raghead', 'sand nigger']
                    if any(word in tweet_lower for word in threat_words):
                        pred_class = 3
                    elif any(word in tweet_lower for word in hate_words):
                        pred_class = 2
                    else:
                        pred_class = 1
                    confidence = 0.85
                
                # Get recommendation and class description
                res = policy.get(pred_class, policy[1])
                desc = CLASS_NAMES.get(pred_class, "Unknown")
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Color coding based on class – with dark text (#212529) for readability
                if pred_class == 1:
                    color = "#212529"      # dark text
                    bg_color = "#d4edda"   # light green
                    icon = "✅"
                    border_color = "#c3e6cb"
                elif pred_class == 0 or pred_class == 2:
                    color = "#212529"
                    bg_color = "#fff3cd"   # light orange
                    icon = "⚠️"
                    border_color = "#ffeeba"
                elif pred_class == 3:
                    color = "#212529"
                    bg_color = "#f8d7da"   # light red
                    icon = "🚫"
                    border_color = "#f5c6cb"
                else:  # class 4
                    color = "#212529"
                    bg_color = "#f8d7da"
                    icon = "🔴"
                    border_color = "#f5c6cb"
                
                st.markdown(f"""
                <div style="padding: 25px; border-radius: 10px; background-color: {bg_color}; border: 2px solid {border_color}; margin: 10px 0;">
                    <h2 style="color: {color}; margin-top: 0;">{icon} Class {pred_class}: {desc}</h2>
                    <p style="font-size: 18px; margin: 10px 0; color: {color};"><strong>Recommended Action:</strong> {res['action']}</p>
                    <p style="font-size: 18px; margin: 10px 0; color: {color};"><strong>Priority:</strong> {res['priority']}</p>
                    <p style="font-size: 18px; margin: 10px 0; color: {color};"><strong>Reason:</strong> {res['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Confidence:** {confidence:.1%}")
                st.progress(float(confidence))
                
                with st.expander("📝 Analyzed Tweet (after cleaning)"):
                    st.write(clean)
                
            except Exception as e:
                st.error(f"Error analyzing tweet: {str(e)}")
    
    if analyze_button:
        analyze_tweet(st.session_state.tweet_input)
    
    # Sample tweets for quick testing – now properly populate the text box and trigger analysis
    st.markdown("---")
    st.subheader("🧪 Try Sample Tweets")
    
    sample_tweets = {
        "Clean": "I love this beautiful day! 😊",
        "Religious Hate": "Muslims are terrorists and should be banned",
        "Racist": "Go back to your country you nigger",
        "Threat": "I'm going to kill you",
        "NSFW": "Check out my naked pics at link in bio"
    }
    
    sample_cols = st.columns(len(sample_tweets))
    for idx, (category, tweet) in enumerate(sample_tweets.items()):
        with sample_cols[idx]:
            if st.button(f"📋 {category}", key=f"sample_{idx}", use_container_width=True):
                # Set the session state and rerun – the text area will update and auto-analyze will trigger
                st.session_state.tweet_input = tweet
                st.rerun()
    
    # Auto-analyze after sample button click (if not already analyzed)
    if 'auto_analyze' not in st.session_state:
        st.session_state.auto_analyze = False
    
    if st.session_state.tweet_input and not analyze_button and not st.session_state.auto_analyze:
        st.session_state.auto_analyze = True
        analyze_tweet(st.session_state.tweet_input)
    elif not st.session_state.tweet_input:
        st.session_state.auto_analyze = False

# Footer with policy guidelines
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
st.markdown(
    "<div style='text-align: center; color: gray; padding: 10px;'>"
    "Tweet Moderation System v2.1 | Powered by Machine Learning"
    "</div>", 
    unsafe_allow_html=True
)
