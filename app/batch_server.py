from fastapi import FastAPI, UploadFile
import httpx
import asyncio
from app.utils import send_to_model

# Queue for incoming requests
request_queue = asyncio.Queue()
BATCH_SIZE = 4   # Process 4 images per batch
BATCH_TIMEOUT = 2  # Seconds

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Start background task for batching
    asyncio.create_task(batch_processor())

@app.get("/")
def root():
    return {"message": "Batch Server running"}

@app.post("/detect")
async def detect(file: UploadFile):
    """
    Accepts single images, adds them to batch queue, 
    returns YOLO detection results when batch is processed.
    """
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await request_queue.put((file, future))
    return await future

async def batch_processor():
    """
    Continuously processes requests in batches.
    """
    while True:
        batch = []
        while len(batch) < BATCH_SIZE:
            try:
                req = await asyncio.wait_for(request_queue.get(), timeout=BATCH_TIMEOUT)
                batch.append(req)
            except asyncio.TimeoutError:
                break

        if batch:
            files = [("files", (f.filename, await f.read(), f.content_type)) for f, _ in batch]
            results = await send_to_model(files)
            for (_, future), result in zip(batch, results["detections"]):
                future.set_result(result)
