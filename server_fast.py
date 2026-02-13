# server_fast.py
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import pandas as pd
import traceback
import os
import uvicorn

# Import the encoder class so unpickling can find it when this script is run as __main__.
# (If your saved pipeline references ml.pipeline_component.VariationResponseEncoder this is not strictly necessary,
# but importing it here is harmless and safe.)
from ml.pipeline_component import VariationResponseEncoder

# Import your pipeline helpers
from ml.pipeline_service import load_pipeline, create_processed_data, single_inference

DEFAULT_PIPELINE_PATH = os.path.join("ml", "pipeline.joblib")

def create_app(pipeline=None):
    app = FastAPI()
    app.state.pipeline = pipeline

    @app.get("/health")
    def health():
        return {"status":"ok"}
    
    @app.post("/single_predict")
    def single_predict(payload: dict = Body(...)):
        """
        Accepts JSON payload:
        - {"Gene": [...], "Variation":[...], "Text":[...]}  OR
        - {"Gene":"g", "Variation":"v", "Text":"t"}  (single sample)
        - OR list of dicts: [{"Gene":"g1","Variation":"v1","Text":"t1"}, ...]
        Returns:
        - list-of-records: [{"ID":..., "pred_class":..., "prob_class_1":..., ...}, ...]
        """
        # parse & normalize input -> query dict-of-lists with keys Gene, Variation, Text
        try:
            data = payload
            if not isinstance(data, (dict, list)):
                raise HTTPException(status_code=400, detail="JSON body must be an object or list")

            if isinstance(data, dict) and all(k in data for k in ("Gene", "Variation", "Text")):
                # dict-of-lists or single sample dict
                if isinstance(data["Gene"], list):
                    query = data
                else:
                    query = {
                        "Gene": [data["Gene"]],
                        "Variation": [data["Variation"]],
                        "Text": [data["Text"]],
                    }
            elif isinstance(data, list):
                # list-of-dicts -> convert to dict-of-lists
                query = {"Gene": [], "Variation": [], "Text": []}
                for rec in data:
                    # tolerate missing keys by using empty string
                    query["Gene"].append(rec.get("Gene", ""))
                    query["Variation"].append(rec.get("Variation", ""))
                    query["Text"].append(rec.get("Text", ""))
            else:
                raise HTTPException(status_code=400, detail="Unexpected JSON format. Expect keys: Gene, Variation, Text")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error parsing input JSON: {exc}")

        # ensure pipeline available (lazy-load fallback)
        pipeline = getattr(app.state, "pipeline", None)
        if pipeline is None:
            try:
                pipeline = load_pipeline(DEFAULT_PIPELINE_PATH)
                app.state.pipeline = pipeline
            except Exception as exc:
                # return 500 with safe message
                raise HTTPException(status_code=500, detail=f"Failed to load model pipeline: {exc}")

        # Build DataFrame and preprocess
        try:
            df = pd.DataFrame(query)
            processed_df = create_processed_data(df)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to prepare input data: {exc}")

        # Run prediction
        try:
            out_df = single_inference(pipeline, query=processed_df)
            result = out_df.to_dict(orient="records")
            return JSONResponse(content=result)
        except Exception as exc:
            # don't leak full trace in production; here we include short message
            tb = traceback.format_exc()
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}\n{tb}")
        
    return app

if __name__ == "__main__":
    pipeline = load_pipeline("ml/pipeline.joblib")
    app = create_app(pipeline=pipeline)
    uvicorn.run(app, host="127.0.0.1", port=5001)
