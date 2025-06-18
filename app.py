import streamlit as st
import transformers 
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Set page config
st.set_page_config(
    page_title="Twitter US Airline Sentiment Analysis",
    page_icon="✈️",
    layout="centered"
)

# Add title and description
st.title("Twitter US Airline Sentiment Analysis")
st.write("Enter a tweet to analyze its sentiment using BERT model")
st.write(f"Transformers version: {transformers.__version__}")

# Load tokenizer dan model BERT yang telah dilatih untuk 3 kelas
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("saved_model")
    model = BertForSequenceClassification.from_pretrained("saved_model", num_labels=3)
    model.eval()
    
    # Gunakan GPU jika tersedia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# Load model
tokenizer, model, device = load_model()

# Fungsi untuk preprocess teks
def preprocess_bert(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    return {key: val.to(device) for key, val in encoded_input.items()}

# Create text input
text_input = st.text_area("Enter your tweet:", height=100)

# Create submit button
if st.button("Analyze Sentiment"):
    if text_input:
        # Preprocess input
        inputs = preprocess_bert(text_input)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        
        # Get sentiment labels and probabilities
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        pred_index = np.argmax(probs)
        prediction = sentiment_labels[pred_index]
        
        # Display results
        st.subheader("Results")
        
        # Create columns for probabilities
        cols = st.columns(3)
        
        # Display each sentiment probability in a column
        for i, (col, label) in enumerate(zip(cols, sentiment_labels)):
            with col:
                st.metric(
                    label=label,
                    value=f"{probs[i]*100:.2f}%",
                    delta=None
                )
        
        # Display overall prediction
        st.success(f"Overall Sentiment: {prediction}")
    else:
        st.warning("Please enter a tweet to analyze.")
