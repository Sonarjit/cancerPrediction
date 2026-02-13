# ml/flask_app.py
import io
from flask import Flask, app, request, jsonify, make_response
from typing import Any, Dict, List
import pandas as pd
from ml.pipeline_component import VariationResponseEncoder
from ml.pipeline_service import create_processed_data, load_pipeline, single_inference, batch_inference

def create_app(pipeline: Any = None) -> Flask:
    app = Flask(__name__)
    app.config["PIPELINE"] = pipeline

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/single_predict", methods=["POST"])
    def single_predict():
        """
        Accepts JSON payload:
         - {"Gene": [...], "Variation":[...], "Text":[...]}  OR
         - {"Gene":"g", "Variation":"v", "Text":"t"}  (single sample)
        Returns:
         - [{"ID":..., "pred_class":..., "prob_class_1":..., ...}, ...]
        """
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "no JSON body"}), 400

        # Normalize to dict-of-lists
        try:
            if isinstance(data, dict) and all(k in data for k in ("Gene", "Variation", "Text")):
                # either each key maps to a list OR to a single value
                if isinstance(data["Gene"], list):
                    query = data
                else:
                    # single sample -> convert to dict-of-lists
                    query = {
                        "Gene": [data["Gene"]],
                        "Variation": [data["Variation"]],
                        "Text": [data["Text"]],
                    }
            elif isinstance(data, list):
                # list-of-dicts
                query = { "Gene": [], "Variation": [], "Text": [] }
                for rec in data:
                    query["Gene"].append(rec.get("Gene", ""))
                    query["Variation"].append(rec.get("Variation", ""))
                    query["Text"].append(rec.get("Text", ""))
            else:
                return jsonify({"error": "Unexpected JSON format"}), 400

            pipeline = app.config.get("PIPELINE")
            if pipeline is None:
                # attempt to lazy-load pipeline (fallback)
                pipeline = load_pipeline()
                app.config["PIPELINE"] = pipeline

            # build DataFrame and preprocess using same function as local code
            
            df = pd.DataFrame(query)
            processed_df = create_processed_data(df)
            out_df = single_inference(pipeline, query=processed_df)
            result = out_df.to_dict(orient="records")
            return jsonify(result)
        except Exception as e:
            # avoid leaking internal large trace; still convey message
            return jsonify({"error": str(e)}), 500

    @app.route("/batch_predict", methods=["POST"])
    def batch_predict():
        """
        Accepts:
        - multipart form upload with key 'file' (CSV)
        - optional form field 'return_csv' (true/false) to return CSV file instead of JSON
        CSV must contain columns: ID, Gene, Variation, Text

        Returns:
        - JSON: {"result": { "ID": [...], "predicted_class": [...], "class1_prob": [...], ... }}
        - OR if return_csv=true: returns CSV attachment (predictions CSV)
        """
        # 1) Get uploaded file
        uploaded = request.files.get("file")
        if uploaded is None:
            return jsonify({"error": "no file uploaded. Send multipart/form-data with key 'file'"}), 400

        # 2) Read CSV into DataFrame (Pandas can read file-like objects)
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            return jsonify({"error": f"failed to read CSV: {exc}"}), 400

        # 3) Basic validation of required columns
        required_cols = {"ID", "Gene", "Variation", "Text"}
        if not required_cols.issubset(df.columns):
            return jsonify({"error": f"CSV must contain columns: {required_cols}"}), 400

        # 4) Ensure pipeline is available (use injected pipeline or lazy-load)
        pipeline = app.config.get("PIPELINE")
        if pipeline is None:
            try:
                pipeline = load_pipeline()    # from prediction_batch.py
                app.config["PIPELINE"] = pipeline
            except Exception as exc:
                return jsonify({"error": f"failed to load pipeline: {exc}"}), 500
        # 5) Preprocess and predict
        try:
            # reuse your preprocessing that expects a DataFrame with ID/Gene/Variation/Text
            processed_df = create_processed_data(df[["ID", "Gene", "Variation", "Text"]])
            df_out = batch_inference(pipeline, query=processed_df, out_csv=None)
        except Exception as exc:
            return jsonify({"error": f"prediction error: {exc}"}), 500
        
        # 6) Handle empty-string sentinel from predict_with_pipeline
        if isinstance(df_out, str):
            if df_out == "":
                # empty result (per existing logic): return an empty JSON structure
                return jsonify({"result": {}}), 200
            else:
                return jsonify({"error": "predict_with_pipeline returned unexpected string"}), 500

        # 7) Convert DataFrame -> JSON structure matching predict_from_csv
        try:
            result = {}
            result["ID"] = df_out["ID"].tolist()
            # ensure integer-typed class labels if possible
            try:
                result["predicted_class"] = df_out["pred_class"].astype(int).tolist()
            except Exception:
                result["predicted_class"] = df_out["pred_class"].tolist()

            prob_cols = [c for c in df_out.columns if c.startswith("prob_class_")]
            prob_cols_sorted = sorted(prob_cols, key=lambda x: int(x.split("_")[-1]))
            for idx, col in enumerate(prob_cols_sorted, start=1):
                key = f"class{idx}_prob"
                result[key] = df_out[col].astype(float).tolist()
        except Exception as exc:
            return jsonify({"error": f"failed to marshal predictions: {exc}"}), 500

        # 8) Optionally return CSV file if requested
        return_csv_flag = request.form.get("return_csv", "false").lower() in ("1", "true", "yes")
        if return_csv_flag:
            # create CSV in-memory and return as attachment
            buf = io.StringIO()
            df_out.to_csv(buf, index=False)
            csv_data = buf.getvalue()
            response = make_response(csv_data)
            response.headers["Content-Disposition"] = "attachment; filename=predictions_batch.csv"
            response.mimetype = "text/csv"
            return response

        # Default: return JSON
        return jsonify({"result": result}), 200

    @app.route("/shutdown", methods=["POST"])
    def shutdown():
        # dev server only
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            return jsonify({"error": "not running with the Werkzeug server"}), 500
        func()
        return jsonify({"status": "shutting down"})

    return app

if __name__ == "__main__":
     # Load pipeline once and pass to flask app
    pipeline = load_pipeline("ml/pipeline.joblib")  # adjust path if needed
    flask_app = create_app(pipeline=pipeline)
    flask_app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)