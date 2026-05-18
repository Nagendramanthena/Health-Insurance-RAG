# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV RUN_MONOLITH=true
ENV USE_NGINX=true

# Disable Mem0 telemetry
ENV MEM0_TELEMETRY=false

# Install system dependencies (Graphviz for diagrams, Nginx for reverse proxy)
RUN apt-get update && apt-get install -y \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    build-essential \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create runtime directories and make everything writable for non-root user 1000 (Hugging Face requirement)
RUN mkdir -p logs storage && chmod -R 777 /app

# Expose the port Hugging Face expects (7860)
EXPOSE 7860

# Run uvicorn backend + nginx reverse proxy
# Streamlit frontend is launched as a sidecar by FastAPI startup event
CMD ["sh", "-c", "\
  mkdir -p /tmp/nginx_client_body /tmp/nginx_proxy /tmp/nginx_fastcgi /tmp/nginx_uwsgi /tmp/nginx_scgi && \
  python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 & \
  nginx -c /app/nginx.conf -g 'daemon off;'"]
