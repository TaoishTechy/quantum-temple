from fastapi import FastAPI, Response
from pathlib import Path
from metrics_prom_buf import latest_metrics

app = FastAPI()

@app.get("/ontology/validate")
def validate():
    # dummy ok for now (engine should update shared state)
    return {"status": "ok", "alignment": latest_metrics.get("alignment", 0.0)}

@app.get("/metrics")
def metrics():
    return Response(content=latest_metrics["raw"], media_type="text/plain; version=0.0.4")
