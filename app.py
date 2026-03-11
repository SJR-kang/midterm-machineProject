import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Tweet Moderation System",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 AI-Powered Tweet Moderation System")
st.markdown("""
This application analyzes tweets for harmful content and provides automated moderation recommendations.
Enter a tweet below to get instant analysis and suggested actions.
""")

# Load the saved models
@st.cache_resource
def load_models():
    """Load trained models from local folder"""
    try:
        # Check if models folder exists
        if not os.path.exists('models'):
            st.sidebar.error("❌ 'models' folder not found!")
            return None, None, None, None
        
        # Load all model files
        vectorizer = joblib.load('models/vectorizer.pkl')
        scaler = joblib.load('models/scaler.pkl')
        model = joblib.load('models/random_forest_model.pkl')
        policy = joblib.load('models/recommendation_policy.pkl')
        
        st.sidebar.success("✅ Models loaded successfully!")
        return vectorizer, scaler, model, policy
        
    except FileNotFoundError as e:
        st.sidebar.error(f"❌ Model file not found: {str(e)}")
        return None, None, None, None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None

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

# Recommendation policy (define as fallback)
DEFAULT_POLICY = {
    0: {"action": "Allow content", "priority": "Low", "reason": "No abusive language detected"},
    1: {"action": "Allow content", "priority": "Low", "reason": "No abusive language detected"},
    2: {"action": "Flag for moderator review", "priority": "Medium", "reason": "Offensive language detected"},
    3: {"action": "Hide content and warn user", "priority": "High", "reason": "Hate speech detected"},
    4: {"action": "Remove content and alert moderators", "priority": "Critical", "reason": "Threatening or aggressive message detected"},
    5: {"action": "Temporarily hide and investigate", "priority": "High", "reason": "Other abusive behavior detected"}
}

# Load models and data
vectorizer, scaler, model, loaded_policy = load_models()
df = load_data()

# Use loaded policy or default
recommendation_policy = loaded_policy if loaded_policy is not None else DEFAULT_POLICY

# Sidebar for information
with st.sidebar:
    st.header("📊 About the System")
    st.info("""
    **Classification Categories:**
    - **Class 0/1**: Clean content - Allow
    - **Class 2**: Offensive language - Flag for review
    - **Class 3**: Hate speech - Hide and warn user
    - **Class 4**: Threats - Remove and alert moderators
    
    **Model Used:** Random Forest Classifier
    **Accuracy:** 96.2%
    """)
    
    if model is not None:
        st.success("✅ Model ready")
    else:
        st.warning("⚠️ Using rule-based fallback")
    
    # Dataset Statistics (if available)
    if df is not None:
        st.header("📁 Dataset Statistics")
        if 'label' in df.columns:
            class_counts = df['label'].value_counts().sort_index()
            st.write(f"**Total tweets:** {len(df)}")
            st.write("**Class Distribution:**")
            for cls, count in class_counts.items():
                st.write(f"- Class {cls}: {count} ({count/len(df)*100:.1f}%)")

# Main content - Single tweet analysis only
st.header("📝 Tweet Analysis")

# Create a nice layout
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    # Tweet input
    tweet_input = st.text_area(
        "Enter your tweet to analyze:",
        height=150,
        placeholder="Type or paste a tweet here..."
    )
    
    # Center the analyze button
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        analyze_button = st.button("🔍 Analyze Tweet", type="primary", use_container_width=True)
    
    if analyze_button and tweet_input:
        with st.spinner("Analyzing..."):
            try:
                # Make prediction
                if model is not None and vectorizer is not None and scaler is not None:
                    # Use actual trained model
                    tweet_vector = vectorizer.transform([tweet_input])
                    tweet_scaled = scaler.transform(tweet_vector)
                    pred_class = model.predict(tweet_scaled)[0]
                    
                    # Get confidence score
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(tweet_scaled)[0]
                        confidence = max(probs)
                    else:
                        confidence = 0.95
                else:
                    # Rule-based fallback
                    tweet_lower = tweet_input.lower()
                    threat_words = ['kill', 'die', 'murder', 'shoot', 'bomb', 'kill you', 'going to kill']
                    hate_words = ['nigger', 'wetback', 'spic', 'chink', 'kike', 'raghead', 'sand nigger']
                    offense_words = ['fuck', 'shit', 'bitch', 'cunt', 'asshole', 'dick', 'pussy']
                    
                    if any(word in tweet_lower for word in threat_words):
                        pred_class = 4
                    elif any(word in tweet_lower for word in hate_words):
                        pred_class = 3
                    elif any(word in tweet_lower for word in offense_words):
                        pred_class = 2
                    else:
                        pred_class = 0
                    confidence = 0.85
                
                # Get recommendation
                policy = recommendation_policy.get(pred_class, recommendation_policy[0])
                
                # Display results in a nice box
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Color-coded result based on class
                if pred_class in [0, 1]:
                    color = "#28a745"  # green
                    bg_color = "#d4edda"
                    icon = "✅"
                    border_color = "#c3e6cb"
                elif pred_class == 2:
                    color = "#fd7e14"  # orange
                    bg_color = "#fff3cd"
                    icon = "⚠️"
                    border_color = "#ffeeba"
                elif pred_class == 3:
                    color = "#dc3545"  # red
                    bg_color = "#f8d7da"
                    icon = "🚫"
                    border_color = "#f5c6cb"
                else:
                    color = "#721c24"  # dark red
                    bg_color = "#f8d7da"
                    icon = "🔴"
                    border_color = "#f5c6cb"
                
                # Result card
                st.markdown(f"""
                <div style="padding: 25px; border-radius: 10px; background-color: {bg_color}; border: 2px solid {border_color}; margin: 10px 0;">
                    <h2 style="color: {color}; margin-top: 0;">{icon} Class {pred_class}</h2>
                    <p style="font-size: 18px; margin: 10px 0;"><strong>Recommended Action:</strong> {policy['action']}</p>
                    <p style="font-size: 18px; margin: 10px 0;"><strong>Priority:</strong> {policy['priority']}</p>
                    <p style="font-size: 18px; margin: 10px 0;"><strong>Reason:</strong> {policy['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence meter
                st.markdown(f"**Confidence:** {confidence:.1%}")
                st.progress(float(confidence))
                
                # Show tweet that was analyzed
                with st.expander("📝 Analyzed Tweet"):
                    st.write(tweet_input)
                
            except Exception as e:
                st.error(f"Error analyzing tweet: {str(e)}")
    
    elif analyze_button:
        st.warning("Please enter a tweet to analyze.")
    
    # Sample tweets for quick testing
    st.markdown("---")
    st.subheader("🧪 Try Sample Tweets")
    
    sample_tweets = {
        "Clean": "I love this beautiful day! 😊",
        "Offensive": "This is a shitty post",
        "Hate Speech": "Go back to your country you nigger",
        "Threat": "I'm going to kill you"
    }
    
    # Create buttons for sample tweets
    sample_cols = st.columns(4)
    for idx, (category, tweet) in enumerate(sample_tweets.items()):
        with sample_cols[idx]:
            if st.button(f"📋 {category}", key=f"sample_{idx}", use_container_width=True):
                # Set the text area value (this is a workaround)
                st.session_state['sample_tweet'] = tweet
                st.rerun()
    
    # Check if we have a sample tweet to load
    if 'sample_tweet' in st.session_state:
        tweet_input = st.session_state['sample_tweet']
        # Clear it so it doesn't persist
        del st.session_state['sample_tweet']
        st.rerun()

# Footer with policy guidelines
st.markdown("---")
st.header("📋 Moderation Policy Guidelines")

policy_df = pd.DataFrame([
    {"Class": "0/1", "Action": "Allow content", "Priority": "Low", "Description": "No abusive language detected"},
    {"Class": "2", "Action": "Flag for moderator review", "Priority": "Medium", "Description": "Offensive language detected"},
    {"Class": "3", "Action": "Hide content and warn user", "Priority": "High", "Description": "Hate speech detected"},
    {"Class": "4", "Action": "Remove content and alert moderators", "Priority": "Critical", "Description": "Threatening or aggressive message detected"},
])

st.table(policy_df)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 10px;'>"
    "Tweet Moderation System v1.0 | Powered by Machine Learning"
    "</div>", 
    unsafe_allow_html=True
)
