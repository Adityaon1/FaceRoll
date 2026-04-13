FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential cmake \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    python3-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p known_faces uploads static

EXPOSE 5000
CMD ["gunicorn", "app:app", "--timeout", "120", "--workers", "1", "--bind", "0.0.0.0:5000"]
