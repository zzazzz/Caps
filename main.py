from typing import List
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile
from random import randint
import os
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from ultralytics import YOLO

IMAGEDIR = "ban-udah-bang-1/train/images/"
app = FastAPI()
model = YOLO('yolov8n.pt')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    image = await file.read()
    image = Image.open(BytesIO(image)).convert("RGB")
    image = np.array(image)

    # Perform object detection with YOLOv8
    detections = model(image)

    return {"detections": detections}

@app.get("/show/")
async def read_random_file():
    files = os.listdir(IMAGEDIR)
    random_index = randint(0, len(files) - 1)
    path = f"{IMAGEDIR}{files[random_index]}"
    return FileResponse(path)
