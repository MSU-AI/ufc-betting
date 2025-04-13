import sys
import os
from flask import Flask, jsonify
import joblib
from results_insert import insert_results
from generate_ev import run_inference_pipeline
from model import FFNClassifier, load_neural_network_model


def create_app():
    app = Flask(__name__)

    # Load the model and scaler
    model_path = os.path.join(os.path.dirname(__file__), "best_model_nn.pt")
    scaler_path = os.path.join(os.path.dirname(__file__), "model_scaler.joblib")

    try:
        app.model = load_neural_network_model(model_path)
        app.scaler = joblib.load(scaler_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model load failed: {e}")
        raise

    @app.route("/")
    def health_check():
        return "Inference API is up"

    @app.route("/inference", methods=["POST"])
    def inference():
        try:
            results = run_inference_pipeline(app.model, app.scaler)
            insert_results(results)
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
