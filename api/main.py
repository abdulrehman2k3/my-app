from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import numpy as np

from PIL import Image
import io
app = FastAPI(
    title="MNIST ANN API",
    description="Handwritten Digit Recognition API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = tf.keras.models.load_model(
    r"D:\Data Science & Ai\Project\model\mnist_ann.keras"
)

@app.get("/")
def home():

    return {
        "message": "MNIST ANN API is running successfully!"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    image = image.convert("L")

    image = image.resize((28, 28))

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_digit = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction)
    )


    return {
        "prediction": predicted_digit,
        "confidence": confidence
    }