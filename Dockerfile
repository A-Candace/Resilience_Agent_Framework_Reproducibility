FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl nginx supervisor libglib2.0-0 libsm6 libxext6 libxrender1 libkrb5-3 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .
COPY deployment/nginx/default.conf /etc/nginx/conf.d/default.conf
RUN rm -f /etc/nginx/sites-enabled/default && mkdir -p /app/model_cache
ENV PYTHONPATH=/app:/app/src STREAMLIT_SERVER_HEADLESS=true
EXPOSE 8080
CMD ["/usr/bin/supervisord", "-c", "/app/deployment/supervisor/supervisord.conf"]
