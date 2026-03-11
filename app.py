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

# Initialize session state for tweet input
if 'tweet_input' not in st.session_state:
    st.session_state.tweet_input = ""

# Title and description
st.title("🔍 AI-Powered Tweet Moderation System")
st.markdown("""
This application analyzes tweets for harmful content and provides automated moderation recommendations.
Enter a tweet below to get instant analysis and suggested actions.
""")

# Function to find model files in different possible locations
@st.cache_resource
def find_and_load_models():
    """Try to find and load models from different possible paths"""
    
    # List of possible paths to check
    possible_model_paths = [
        'models/',                          # models folder in current dir
        './models/',                        # same as above
        '../models/',                       # one level up
        '/mount/src/midterm-machineproject/models/',  # Streamlit Cloud path
        '',                                  # current directory
    ]
    
    possible_filenames = {
        'vectorizer': ['vectorizer.pkl', 'vectorizer.joblib'],
        'scaler': ['scaler.pkl', 'scaler.joblib'],
        'model': ['random_forest_model.pkl', 'random_forest_model.joblib', 'model.pkl'],
        'policy': ['recommendation_policy.pkl', 'policy.pkl']
    }
    
    # Try each path
    for base_path in possible_model_paths:
        try:
            if not os.path.exists(base_path):
                continue
                
            st.sidebar.write(f"Checking: {base_path}")
            
            # Try to load each file with different possible names
            vectorizer = None
            scaler = None
            model = None
            policy = None
            
            # Try vectorizer
            for filename in possible_filenames['vectorizer']:
                full_path = os.path.join(base_path, filename)
                if os.path.exists(full_path):
                    vectorizer = joblib.load(full_path)
                    st.sidebar.write(f"✅ Found vectorizer: {full_path}")
                    break
            
            # Try scaler
            for filename in possible_filenames['scaler']:
                full_path = os.path.join(base_path, filename)
                if os.path.exists(full_path):
                    scaler = joblib.load(full_path)
                    st.sidebar.write(f"✅ Found scaler: {full_path}")
                    break
            
            # Try model
            for filename in possible_filenames['model']:
                full_path = os.path.join(base_path, filename)
                if os.path.exists(full_path):
                    model = joblib.load(full_path)
                    st.sidebar.write(f"✅ Found model: {full_path}")
                    break
            
            # Try policy
            for filename in possible_filenames['policy']:
                full_path = os.path.join(base_path, filename)
                if os.path.exists(full_path):
                    policy = joblib.load(full_path)
                    st.sidebar.write(f"✅ Found policy: {full_path}")
                    break
            
            # If we found all files, return them
            if all([vectorizer, scaler, model, policy]):
                return vectorizer, scaler, model, policy
                
        except Exception as e:
            st.sidebar.write(f"Error with path {base_path}: {str(e)}")
            continue
    
    # If we get here, no models found
    return None, None, None, None

# Recommendation policy (define as fallback)
DEFAULT_POLICY = {
    0: {"action": "Allow content", "priority": "Low", "reason": "No abusive language detected"},
    1: {"action": "Allow content", "priority": "Low", "reason": "No abusive language detected"},
    2: {"action": "Flag for moderator review", "priority": "Medium", "reason": "Offensive language detected"},
    3: {"action": "Hide content and warn user", "priority": "High", "reason": "Hate speech detected"},
    4: {"action": "Remove content and alert moderators", "priority": "Critical", "reason": "Threatening or aggressive message detected"},
    5: {"action": "Temporarily hide and investigate", "priority": "High", "reason": "Other abusive behavior detected"}
}

# Load models with better error handling
vectorizer, scaler, model, loaded_policy = find_and_load_models()

# Use loaded policy or default
recommendation_policy = loaded_policy if loaded_policy is not None else DEFAULT_POLICY

# Load data for statistics (optional)
@st.cache_data
def load_data():
    """Load data for statistics display"""
    try:
        possible_paths = [
            'Tweets_reclassified.csv', 
            'Tweets.csv', 
            'data/Tweets.csv',
            './Tweets_reclassified.csv',
            '../Tweets_reclassified.csv'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Loaded data from {path}")
                return df
        return None
    except Exception as e:
        st.sidebar.write(f"Data loading error: {str(e)}")
        return None

df = load_data()

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
    """)
    
    if model is not None:
        st.success("✅ Model ready")
        if hasattr(model, 'get_params'):
            st.write(f"Model type: {type(model).__name__}")
    else:
        st.warning("⚠️ Using rule-based fallback")
        st.info("""
        To use the actual ML model:
        1. Create a 'models' folder
        2. Add your .pkl files:
           - vectorizer.pkl
           - scaler.pkl  
           - random_forest_model.pkl
           - recommendation_policy.pkl
        """)
    
    # Dataset Statistics (if available)
    if df is not None:
        st.header("📁 Dataset Statistics")
        st.write(f"**Total tweets:** {len(df)}")
        if 'label' in df.columns:
            class_counts = df['label'].value_counts().sort_index()
            st.write("**Class Distribution:**")
            for cls, count in class_counts.items():
                st.write(f"- Class {cls}: {count} ({count/len(df)*100:.1f}%)")

# Main content - Single tweet analysis only
st.header("📝 Tweet Analysis")

# Create a nice layout
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    # Tweet input - using session state
    tweet_input = st.text_area(
        "Enter your tweet to analyze:",
        height=150,
        placeholder="Type or paste a tweet here...",
        value=st.session_state.tweet_input,
        key="tweet_input_area"
    )
    
    # Update session state when user types
    st.session_state.tweet_input = tweet_input
    
    # Center the analyze button
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        analyze_button = st.button("🔍 Analyze Tweet", type="primary", use_container_width=True)
    
    # Function to analyze tweet
    def analyze_tweet(tweet_text):
        if not tweet_text:
            st.warning("Please enter a tweet to analyze.")
            return
        
        with st.spinner("Analyzing..."):
            try:
                # Make prediction
                if model is not None and vectorizer is not None and scaler is not None:
                    # Use actual trained model
                    tweet_vector = vectorizer.transform([tweet_text])
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
                    tweet_lower = tweet_text.lower()
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
                    st.write(tweet_text)
                
            except Exception as e:
                st.error(f"Error analyzing tweet: {str(e)}")
    
    # Handle analyze button click
    if analyze_button:
        analyze_tweet(st.session_state.tweet_input)
    
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
                st.session_state.tweet_input = tweet
                st.rerun()
    
    # Auto-analyze if there's a tweet from sample button
    if st.session_state.tweet_input and st.session_state.tweet_input != "":
        if 'last_analyzed' not in st.session_state or st.session_state.last_analyzed != st.session_state.tweet_input:
            st.session_state.last_analyzed = st.session_state.tweet_input
            analyze_tweet(st.session_state.tweet_input)

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
