# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV RUN_MONOLITH=true
ENV USE_NGINX=true

# Install system dependencies (including Graphviz for diagrams and Nginx for reverse proxy)
RUN apt-get update && apt-get install -y \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    build-essential \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create logs and storage directories
RUN mkdir -p logs storage

# Expose the port Hugging Face expects (7860)
EXPOSE 7860

# Run both uvicorn backend and nginx reverse proxy
CMD ["sh", "-c", "mkdir -p /tmp/nginx_client_body /tmp/nginx_proxy /tmp/nginx_fastcgi /tmp/nginx_uwsgi /tmp/nginx_scgi && python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 & nginx -c /app/nginx.conf -g 'daemon off;'"]
