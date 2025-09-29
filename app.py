# Importing dependencies

import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Mapping to word index ( for understanding )

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key, value in word_index.items()}

# Load model

model = load_model('data/simple_rnn_imdb.h5')

# Helper function - decode review and preprocess text

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen = 500)
    return padded_review

# Prediction function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit app

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Your movie review will be classified as positive or negative.')

user_input = st.text_input('Enter a movie review : ')

if st.button('Classify'):
    sentiment, prediction = predict_sentiment(user_input)
    st.write('Review :- ', sentiment)
    st.write('Confidence Score :- ', prediction)
else:
    st.write('You did not enter a movie review.')
