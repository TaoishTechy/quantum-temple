# FILE: quantum_temple_multiverse/infra/Dockerfile
FROM python:3.11-slim
ENV PYTHONHASHSEED=0
ENV OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["python","-m","quantum_temple_multiverse.multiverse.core"]
