from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware


Model= tf.keras.models.load_model("../models/1")
class_names=["Early Blight","Late Blight","Healthy"]


app=FastAPI()
origins=[
    "http://localhost",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]

)
@app.get("/ping")
async def ping():
    return "Server is alive"

def read_file_as_image(data)-> np.ndarray:
    return np.array(Image.open(BytesIO(data)))

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    img=read_file_as_image(await file.read())
    img_batch=np.expand_dims(img,0)
    pred=Model.predict(img_batch)
    res=class_names[np.argmax(pred[0])]
    confidence=np.max(pred[0])
    return {'class':res,
            'confidence': float(confidence)
            }
    pass

if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8000)