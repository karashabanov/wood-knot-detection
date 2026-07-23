FROM python:3.12-slim
WORKDIR /app
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
RUN mkdir -p /tmp/Ultralytics && chmod -R 777 /tmp/Ultralytics
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app