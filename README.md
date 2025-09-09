# YOLOv8 FastAPI Batch Inference Server

This project provides a **scalable batch inference pipeline** for YOLOv8 using **FastAPI**.  

It is split into two services:
1. **Batch Server** – Receives single image requests, queues them, and groups them into batches before sending them to the model server.
2. **Model Server** – Loads the YOLOv8 model, performs inference on a batch of images, and returns structured detections.

---

## Features
- **Batching support**: multiple single-image requests are combined into batches to optimize GPU usage.
- **YOLOv8 detection**: runs on the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) model.
- **FastAPI endpoints**: clean, async APIs for inference.
- **Dockerized**: ready to run inside containers.
- **Graceful startup/shutdown**: YOLO model loads on startup and cleans up on shutdown.

---

## Installation (Local)

### 1. Clone the repository
```bash
git clone https://github.com/your-repo/yolo-batch-server.git
cd yolo-batch-server
```
## Create and activate a virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
## Install Dependencies 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
## Requirements

- Python 3.9+
- Ultralytics YOLO
- FastAPI
- httpx
- Pillow
- Uvicorn (for serving)
