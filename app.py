from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

app = Flask(__name__)

# Load tokenizer dan model BERT yang telah dilatih untuk 3 kelas
tokenizer = BertTokenizer.from_pretrained("saved_model")
model = BertForSequenceClassification.from_pretrained("saved_model", num_labels=3)
model.eval()

# Gunakan GPU jika tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fungsi untuk preprocess teks
def preprocess_bert(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    return {key: val.to(device) for key, val in encoded_input.items()}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probabilities = None
    text = ""

    if request.method == "POST":
        text = request.form["text"]
        inputs = preprocess_bert(text)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        pred_index = np.argmax(probs)
        prediction = sentiment_labels[pred_index]
        probabilities = {
            f"{label} ({i})": round(float(probs[i]) * 100, 2)
            for i, label in enumerate(sentiment_labels)
        }

    return render_template("index.html", prediction=prediction, probabilities=probabilities, text=text)

if __name__ == "__main__":
    app.run(debug=True)
