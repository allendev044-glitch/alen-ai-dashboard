import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os
import ssl

# Mac SSL fix
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up the web page layout
st.set_page_config(page_title="Alen AI", page_icon="🤖")

st.title("🤖 Chat with Alen 1.0")
st.write("Type a sentence or a movie review below, and Alen's LSTM brain will analyze the sentiment!")

# Cache the model so it only loads once when the server starts
@st.cache_resource
def load_alen_model():
    return tf.keras.models.load_model('alen_text_classifier.keras')

# Cache the dictionary so it only loads once
@st.cache_data
def load_dictionary():
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  
    word_index["<UNUSED>"] = 3
    return word_index

# Load them into memory
with st.spinner("Waking up Alen..."):
    alen_model = load_alen_model()
    word_index = load_dictionary()

def encode_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    encoded = [1] 
    for word in text.split():
        if word in word_index and word_index[word] < 10000:
            encoded.append(word_index[word])
        else:
            encoded.append(2)
    return pad_sequences([encoded], maxlen=256, padding='post')

# Build the User Interface
user_input = st.text_area("Enter your text here:", placeholder="I absolutely loved this experience...")

if st.button("Ask Alen"):
    if user_input.strip():
        # Process text and predict
        encoded_sentence = encode_text(user_input)
        
        # Convert the prediction to a standard Python float so Streamlit can read it easily
        prediction = float(alen_model.predict(encoded_sentence, verbose=0)[0][0])
        
        st.divider() 
        st.subheader("Analysis Results")
        
        # Create two columns for a sleek dashboard layout
        col1, col2 = st.columns(2)
        
     # Display the results with a Neutral zone
        if prediction >= 0.6:
            confidence = prediction * 100
            col1.success("Verdict: **POSITIVE**")
            col2.metric("Confidence Score", f"{confidence:.1f}%")
        elif prediction <= 0.4:
            confidence = (1 - prediction) * 100
            col1.error("Verdict: **NEGATIVE**")
            col2.metric("Confidence Score", f"{confidence:.1f}%")
        else:
            col1.warning("Verdict: **NEUTRAL / UNCERTAIN**")
            col2.metric("Raw Score", f"{prediction * 100:.1f}%")
            
        # Add a visual gauge bar (0.0 is entirely negative, 1.0 is entirely positive)
        st.write("Sentiment Gauge (Left = Negative, Right = Positive):")
        st.progress(prediction)
        
    else:
        st.warning("Please type something first!")