from fastapi import FastAPI, APIRouter, UploadFile
from PIL import Image

from app.backend.predict import predict

app = FastAPI()


@app.router.get("/")
def hello():
    return {"respose": "ok"}

@app.router.post("/predict")
async def prediction(userImage: UploadFile):
    predict_image = predict(userImage.file)
    
    return {"prediction": predict_image} 