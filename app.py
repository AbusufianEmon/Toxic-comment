from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import json
import re

# Load model and vocab
model = tf.keras.models.load_model('combined_model.h5')
with open('vocab.json') as f:
    word_index = json.load(f)

# Constants
MAX_LEN = 800
THRESHOLD = 0.4
LABELS = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]

# Tokenize input
def tokenize(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    sequence = [word_index.get(word, 1) for word in words]
    if len(sequence) < MAX_LEN:
        sequence += [0] * (MAX_LEN - len(sequence))
    return sequence[:MAX_LEN]

# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    input_text = ""
    message = ""
    detected_labels = []

    if request.method == 'POST':
        input_text = request.form['comment']
        sequence = tokenize(input_text)
        pred = model.predict(np.array([sequence]))[0]

        # Detect which labels exceeded threshold
        detected_labels = [label for label, score in zip(LABELS, pred) if score > THRESHOLD]

        if detected_labels:
            message = "✅ Yes, this comment is toxic."
        else:
            message = "❌ No, this comment is not toxic."

    return render_template('index.html',
                           result_message=message,
                           comment=input_text,
                           labels=detected_labels)

if __name__ == '__main__':
    app.run(debug=True)
