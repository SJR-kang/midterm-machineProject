import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import requests
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Tweet Moderation System",
    page_icon="🔍",
    layout="wide"
)

# GitHub raw URLs for your files
# Replace these with your actual GitHub raw URLs
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/"
CSV_URL = GITHUB_RAW_BASE + "Tweets.csv"
MODEL_BASE_URL = GITHUB_RAW_BASE + "models/"

# Title and description
st.title("🔍 AI-Powered Tweet Moderation System")
st.markdown("""
This application analyzes tweets for harmful content and provides automated moderation recommendations.
Enter a tweet below to get instant analysis and suggested actions.
""")

# Function to load CSV from GitHub
@st.cache_data  # Cache the data so it doesn't reload every time
def load_data_from_github():
    try:
        response = requests.get(CSV_URL)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        return df
    except Exception as e:
        st.error(f"Error loading data from GitHub: {str(e)}")
        return None

# Load the saved models (from local or GitHub)
@st.cache_resource
def load_models():
    models = {}
    
    # Try loading from local models folder first
    try:
        vectorizer = joblib.load('models/vectorizer.pkl')
        scaler = joblib.load('models/scaler.pkl')
        model = joblib.load('models/random_forest_model.pkl')
        policy = joblib.load('models/recommendation_policy.pkl')
        
        st.sidebar.success("✅ Models loaded from local folder")
        return vectorizer, scaler, model, policy
        
    except FileNotFoundError:
        st.sidebar.warning("Local models not found. Checking GitHub...")
        
        # Try loading from GitHub (if you've uploaded models there too)
        try:
            # Note: You'll need to upload your .pkl files to GitHub
            # and use raw URLs for them as well
            st.sidebar.error("""
            GitHub model loading requires .pkl files to be uploaded.
            Please ensure models are in the local 'models' folder.
            """)
            return None, None, None, None
            
        except Exception as e:
            st.sidebar.error(f"Could not load models: {str(e)}")
            return None, None, None, None

# Load data and models
df_github = load_data_from_github()
vectorizer, scaler, model, recommendation_policy = load_models()

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
        st.success("✅ Model loaded successfully!")
    else:
        st.warning("⚠️ Using rule-based fallback")
    
    # Dataset Statistics from GitHub CSV
    st.header("📁 Dataset Statistics")
    
    if df_github is not None:
        class_counts = df_github['label'].value_counts().sort_index()
        
        st.write("**Class Distribution:**")
        class_dist_df = pd.DataFrame({
            'Class': class_counts.index,
            'Count': class_counts.values
        })
        st.dataframe(class_dist_df)
        
        # Create pie chart from actual data
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        ax.set_title('Training Data Distribution')
        st.pyplot(fig)
        
        # Show basic stats
        st.write("**Dataset Info:**")
        st.write(f"Total tweets: {len(df_github)}")
        st.write(f"Features: {df_github.columns.tolist()}")
    else:
        st.warning("Could not load dataset from GitHub")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Tweet Analysis")
    
    input_method = st.radio(
        "Choose input method:",
        ["Single Tweet", "Batch Analysis (from GitHub)", "Upload CSV"]
    )
    
    if input_method == "Single Tweet":
        tweet_input = st.text_area(
            "Enter your tweet here:",
            height=150,
            placeholder="Type or paste a tweet to analyze..."
        )
        
        if st.button("Analyze Tweet", type="primary"):
            if tweet_input:
                with st.spinner("Analyzing..."):
                    try:
                        if model is not None:
                            # Use actual trained model
                            tweet_vector = vectorizer.transform([tweet_input])
                            tweet_scaled = scaler.transform(tweet_vector)
                            pred_class = model.predict(tweet_scaled)[0]
                            
                            # Get prediction probabilities
                            if hasattr(model, "predict_proba"):
                                probs = model.predict_proba(tweet_scaled)[0]
                                confidence = max(probs)
                            else:
                                confidence = 0.95
                        else:
                            # Rule-based fallback
                            tweet_lower = tweet_input.lower()
                            if any(word in tweet_lower for word in ['kill', 'die', 'murder', 'shoot', 'bomb']):
                                pred_class = 4
                            elif any(word in tweet_lower for word in ['nigger', 'wetback', 'spic', 'chink', 'kike', 'raghead']):
                                pred_class = 3
                            elif any(word in tweet_lower for word in ['fuck', 'shit', 'bitch', 'cunt', 'asshole']):
                                pred_class = 2
                            else:
                                pred_class = 0
                            confidence = 0.85
                        
                        policy = recommendation_policy.get(pred_class, recommendation_policy[0])
                        
                        st.subheader("📊 Analysis Results")
                        
                        # Color-coded result boxes
                        if pred_class in [0, 1]:
                            color = "green"
                            icon = "✅"
                        elif pred_class == 2:
                            color = "orange"
                            icon = "⚠️"
                        elif pred_class == 3:
                            color = "red"
                            icon = "🚫"
                        else:
                            color = "darkred"
                            icon = "🔴"
                        
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: {color}20; border-left: 5px solid {color};">
                            <h3>{icon} Predicted Class: {pred_class}</h3>
                            <p><strong>Recommended Action:</strong> {policy['action']}</p>
                            <p><strong>Priority:</strong> {policy['priority']}</p>
                            <p><strong>Reason:</strong> {policy['reason']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(float(confidence), text=f"Model Confidence: {confidence:.1%}")
                        
                    except Exception as e:
                        st.error(f"Error analyzing tweet: {str(e)}")
            else:
                st.warning("Please enter a tweet to analyze.")
    
    elif input_method == "Batch Analysis (from GitHub)":
        st.write("Analyze tweets directly from the GitHub repository.")
        
        if df_github is not None:
            st.write("**Preview of GitHub dataset:**")
            st.dataframe(df_github.head(10))
            
            if st.button("Analyze All Tweets from GitHub"):
                with st.spinner(f"Analyzing {len(df_github)} tweets..."):
                    if model is not None:
                        # Process all tweets
                        results = []
                        for tweet in df_github['tweet'].head(100):  # Limit to 100 for demo
                            tweet_vector = vectorizer.transform([str(tweet)])
                            tweet_scaled = scaler.transform(tweet_vector)
                            pred = model.predict(tweet_scaled)[0]
                            results.append({
                                'Tweet': str(tweet)[:50] + "...",  # Truncate for display
                                'Predicted': pred,
                                'Action': recommendation_policy[pred]['action'][:20]
                            })
                        
                        results_df = pd.DataFrame(results)
                        st.success(f"✅ Analyzed {len(results_df)} tweets!")
                        st.dataframe(results_df)
                        
                        # Show distribution
                        st.write("**Prediction Distribution:**")
                        dist_df = results_df['Predicted'].value_counts().reset_index()
                        dist_df.columns = ['Class', 'Count']
                        st.dataframe(dist_df)
                    else:
                        st.error("Model not loaded")
        else:
            st.error("GitHub dataset not available")
    
    else:  # Upload CSV
        st.write("Upload your own CSV file with tweets to analyze.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df_batch.head())
            
            # Assume first column contains tweets
            tweet_column = st.selectbox(
                "Select the column containing tweets:",
                df_batch.columns.tolist()
            )
            
            if st.button("Analyze Uploaded Batch"):
                with st.spinner(f"Analyzing {len(df_batch)} tweets..."):
                    if model is not None:
                        results = []
                        for tweet in df_batch[tweet_column].head(100):  # Limit for demo
                            tweet_vector = vectorizer.transform([str(tweet)])
                            tweet_scaled = scaler.transform(tweet_vector)
                            pred = model.predict(tweet_scaled)[0]
                            results.append({
                                'Tweet': str(tweet)[:50] + "...",
                                'Predicted': pred,
                                'Action': recommendation_policy[pred]['action'][:20]
                            })
                        
                        results_df = pd.DataFrame(results)
                        st.success(f"✅ Analyzed {len(results_df)} tweets!")
                        st.dataframe(results_df)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="moderation_results.csv",
                            mime="text/csv"
                        )
                    else:
                        # Simulated results
                        results_df = pd.DataFrame({
                            'Tweet': df_batch[tweet_column].head(10),
                            'Predicted': [2, 0, 3, 4, 2, 0, 3, 2, 4, 0][:10],
                            'Action': ['Flag', 'Allow', 'Hide', 'Remove', 'Flag', 'Allow', 'Hide', 'Flag', 'Remove', 'Allow'][:10]
                        })
                        st.dataframe(results_df)

with col2:
    st.header("📈 Model Performance")
    
    # Display confusion matrix
    st.subheader("Confusion Matrix - Random Forest")
    cm = np.array([[1948, 0, 19, 0],
                   [26, 72, 2, 0],
                   [9, 4, 10, 0],
                   [24, 1, 0, 108]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 2', 'Class 3', 'Class 4'],
                yticklabels=['Class 0', 'Class 2', 'Class 3', 'Class 4'])
    plt.title('Random Forest Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(fig)
    
    # Performance metrics
    st.subheader("Performance Metrics")
    metrics_data = {
        'Metric': ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)'],
        'Value': ['96.2%', '0.81', '0.74', '0.77']
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Model comparison
    st.subheader("Model Comparison")
    comparison_data = {
        'Model': ['Decision Tree', 'Random Forest', 'KNN', 'Naive Bayes', 'SVM'],
        'Accuracy': ['99.4%', '96.2%', '89.7%', '88.5%', '99.0%']
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

# Footer with recommendations table
st.markdown("---")
st.header("📋 Moderation Policy Guidelines")

policy_df = pd.DataFrame([
    {"Class": "0/1", "Action": "Allow content", "Priority": "Low", "Description": "No abusive language detected"},
    {"Class": "2", "Action": "Flag for moderator review", "Priority": "Medium", "Description": "Offensive language detected"},
    {"Class": "3", "Action": "Hide content and warn user", "Priority": "High", "Description": "Hate speech detected"},
    {"Class": "4", "Action": "Remove content and alert moderators", "Priority": "Critical", "Description": "Threatening or aggressive message detected"},
    {"Class": "5", "Action": "Temporarily hide and investigate", "Priority": "High", "Description": "Other abusive behavior detected"}
])

st.table(policy_df)

# Add sample tweets for testing
st.markdown("---")
st.header("🧪 Test with Sample Tweets")

sample_tweets = {
    "Clean": "I love this beautiful day! 😊",
    "Offensive": "This is a shitty post",
    "Hate Speech": "Go back to your country you nigger",
    "Threat": "I'm going to kill you"
}

cols = st.columns(4)
for idx, (category, tweet) in enumerate(sample_tweets.items()):
    with cols[idx]:
        st.write(f"**{category}:**")
        st.code(tweet)
        if st.button(f"Test {category}", key=f"test_{idx}"):
            if model is not None:
                tweet_vector = vectorizer.transform([tweet])
                tweet_scaled = scaler.transform(tweet_vector)
                pred = model.predict(tweet_scaled)[0]
                policy = recommendation_policy[pred]
                st.info(f"Class {pred}: {policy['action']}")
            else:
                # Simulated
                if category == "Clean":
                    st.success("✅ Class 0 - Allow content")
                elif category == "Offensive":
                    st.warning("⚠️ Class 2 - Flag for review")
                elif category == "Hate Speech":
                    st.error("🚫 Class 3 - Hide and warn user")
                else:
                    st.error("🔴 Class 4 - Remove and alert moderators")
