from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

model = load_model('best_stock_model.h5')

@app.route("/")
def home():
    # Get all PNG files in the static folder
    static_folder = os.path.join(app.root_path, 'static')
    graph_files = [f for f in os.listdir(static_folder) if f.endswith('_test_prediction.png')]
    
    # Create a list of dictionaries containing information about each graph
    graphs = [
        {
            'title': f.split('_')[0],  # Assumes filename format is "STOCKSYMBOL_test_prediction.png"
            'filename': f
        } for f in graph_files
    ]
    
    return render_template("index.html", graphs=graphs)

if __name__ == "__main__":
    app.run(debug=True, port=5001)