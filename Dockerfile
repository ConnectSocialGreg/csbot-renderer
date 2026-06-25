FROM python:3.11-slim

# weasyprint system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangocairo-1.0-0 libpangoft2-1.0-0 \
    libcairo2 libgdk-pixbuf-2.0-0 libffi-dev \
    libfontconfig1 libharfbuzz0b fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Render/Railway provide $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
