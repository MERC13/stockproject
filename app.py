import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('best_stock_model.h5')

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)