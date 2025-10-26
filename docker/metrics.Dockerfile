FROM python:3.11-slim
WORKDIR /app
COPY ./docker/metrics_app.py /app/metrics_app.py
RUN pip install --no-cache-dir fastapi uvicorn
EXPOSE 8080
CMD ["uvicorn", "metrics_app:app", "--host", "0.0.0.0", "--port", "8080"]
