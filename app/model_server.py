from fastapi import FastAPI, UploadFile
from contextlib import asynccontextmanager
from ultralytics import YOLO
from PIL import Image
import io

# Model is not loaded globally here
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events:
    - Loads YOLO at startup
    - Cleans up at shutdown
    """
    global model
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded successfully")
    yield
    print("Shutting down model server...")
    del model

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "YOLO Model Server running with startup/shutdown lifecycle"}

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile]):
    """
    Accepts multiple images in one request, runs YOLO detection, and returns results.
    """
    global model
    if model is None:
        return {"error": "Model not loaded yet"}

    # Convert images to PIL
    images = []
    for file in files:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
        images.append(img)

    # Run inference in batch
    results = model(images)

    output = []
    for r in results:
        detections = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            detections.append({
                "class": name,
                "confidence": conf,
                "box": box.xyxy[0].tolist()
            })
        output.append(detections)

    return {"detections": output}
