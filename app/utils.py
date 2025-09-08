import httpx

MODEL_SERVER_URL = "http://model-server:8000/predict_batch"

async def send_to_model(files):
    """
    Sends batched images to the model server for inference.
    """
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(MODEL_SERVER_URL, files=files)
        return response.json()
