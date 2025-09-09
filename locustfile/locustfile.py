from locust import HttpUser, task, between

class YoloUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        # Test request with one image
        with open("IMG_9345.png", "rb") as f:
            self.client.post("/detect", files={"file": f})
