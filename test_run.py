from flask import Flask, render_template, jsonify

import torch, sys, os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model import AADModel
import threading
from utils import preprocess, infer, process_data_stream


app = Flask(__name__)
@app.route("/index")
def dashboard():
    """
    메인 대시보드.
    """
    return render_template("test.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)