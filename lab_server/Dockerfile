# R.A.I.N. Lab — maritime.sh deployment image
# Build context: repo root (docker build -f lab_server/Dockerfile .)

FROM python:3.11-slim

WORKDIR /app

# Install web server deps first (cached layer)
COPY lab_server/requirements.txt lab_server/requirements.txt
RUN pip install --no-cache-dir -r lab_server/requirements.txt

# Install repo runtime deps (run_rain_lab and its imports)
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir \
    openai \
    pyyaml \
    rich \
    requests \
    duckduckgo-search \
    || true

# Copy only what the server needs — not the full repo
COPY rain_lab_runtime.py .
COPY truth_layer.py .
COPY james_library/ james_library/
COPY lab_server/ lab_server/

# Optional: copy papers/library context if present
COPY papers/ papers/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "lab_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
