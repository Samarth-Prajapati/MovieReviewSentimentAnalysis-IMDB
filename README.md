# 🎬 IMDB Movie Review Sentiment Analysis

This project is a **Deep Learning + Streamlit web app** that classifies IMDB movie reviews as **Positive** or **Negative** using a trained **Simple RNN** model.  

The app takes a text review from the user, processes it, and predicts the sentiment along with a confidence score.

---

## 📌 Features
- Input any movie review in plain text
- Sentiment classification: **Positive** / **Negative**
- Confidence score for the prediction
- User-friendly **Streamlit interface**

---

## 🛠️ Tech Stack
- **Python 3.13**
- **TensorFlow / Keras** (Deep Learning)
- **Streamlit** (Web App)
- **NumPy**
- **IMDB Dataset** (built-in Keras dataset)

---

## 🚀 Installation & Setup

```bash
git clone https://github.com/your-username/imdb-sentiment-rnn.git
cd imdb-sentiment-rnn

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

pip install -r requirements.txt

streamlit run app.py
