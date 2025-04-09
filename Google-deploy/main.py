import sys
import os
from flask import Flask, jsonify
import joblib
from results_insert import insert_results
from generate_ev import run_inference_pipeline
from model import FFNClassifier


def create_app():
    app = Flask(__name__)

    # Load the model once at startup
    model_path = os.path.join(os.path.dirname(__file__), "best_model.joblib")
    try:
        app.model = joblib.load(model_path)
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
            results = run_inference_pipeline(app.model)
            insert_results(results)
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
