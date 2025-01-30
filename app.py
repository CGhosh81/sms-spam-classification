import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download necessary NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')

# Initialize objects
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize
    y = [i for i in text if i.isalnum()]  # Remove special characters

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load pre-trained vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    mnb = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Streamlit app
st.title("Email/SMS Spam Classifier")

# Input message
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    try:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = mnb.predict(vector_input)[0]

        # 4. Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
