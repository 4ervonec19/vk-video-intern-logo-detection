FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY project/app.py project/
COPY project/templates/upload.html project/templates/
COPY project/inference_embeddings.py project/
COPY project/inference_text_detection.py project/
COPY notebooks/yolo/yolo_weigths/best.pt notebooks/yolo/yolo_weigths/
COPY notebooks/metric_learning/labels/idx2classname.json notebooks/metric_learning/labels/
COPY notebooks/metric_learning/arcface_weights/best-precision-arcfaceloss-epoch=14-precision_at_1_epoch=0.95.ckpt notebooks/metric_learning/arcface_weights/

COPY .env .env
RUN export $(cat .env | xargs)

WORKDIR /app/project

EXPOSE 8002

CMD ["python", "app.py"]

