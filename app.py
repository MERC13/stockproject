from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model('best_stock_model.h5')

@app.route("/")
def home():
    graphs = [
        {"title": "Stock Price Prediction", "filename": "stock_prediction.png"},
        {"title": "Stock Price History", "filename": "stock_history.png"}
    ]
    return render_template("index.html", graphs=graphs)

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    # Process the input data and use your model to make predictions
    # This is a placeholder implementation
    prediction = model.predict(np.array([[0, 0, 0, 0]]))  # Replace with actual input data
    
    # Generate chart data (placeholder)
    chart_data = {
        "labels": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
        "actual": [100, 102, 98, 103, 105],
        "predicted": [101, 103, 99, 102, 104]
    }
    
    return jsonify({"prediction": float(prediction[0][0]), "chartData": chart_data})

if __name__ == "__main__":
    app.run(debug=True, port=5001)