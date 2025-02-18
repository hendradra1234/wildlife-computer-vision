from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI, File
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import numpy as np
import numpy as np
import requests
from PIL import Image
import json
import io

URL = "http://tf_serving:8501/v1/models/wildlife_computer_vision:predict"
img_size = (224, 224)
labels = [
    'butterfly',
    'cat',
    'chicken',
    'cow',
    'dog',
    'elephant',
    'horse',
    'sheep',
    'spider',
    'squirrel'
]

app = FastAPI()

# title
app = FastAPI(
    title="Wildlife API Inference",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="2025.1.01",
)

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def save_openapi_json():
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")

def preprocess_image(image: UploadFile):
    img = Image.open(io.BytesIO(image.file.read()))
    img = img.resize(img_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img.tolist()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_data = preprocess_image(file)
    payload = {"instances": img_data}
    response = requests.post(URL, json=payload)
    result = response.json()
    print('payload', result)
    raw_predictions = result.get("predictions")[0]
    max_index = int(np.argmax(raw_predictions))

    predicted_label = labels[max_index] if max_index < len(labels) else "unknown"
    confidence = float(raw_predictions[max_index])

    return {"predict": predicted_label, "confidence": confidence}