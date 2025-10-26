from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    # Simple placeholder Prometheus exposition format
    body = (
        "# HELP quantum_plv_total Phase Lock Value\n"
        "# TYPE quantum_plv_total gauge\n"
        "quantum_plv_total 0.91\n"
    )
    return Response(content=body, media_type="text/plain; version=0.0.4")
