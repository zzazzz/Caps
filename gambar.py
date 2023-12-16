from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageDraw
import pandas as pd
from ultralytics import YOLO
import numpy as np
import io
from fastapi.responses import JSONResponse, StreamingResponse


app = FastAPI()
model_driver = YOLO("C:/Users/ziyad/Documents/Bangkit/deployment/best.pt")
model_object = YOLO("C:/Users/ziyad/Documents/Bangkit/deployment/best2.pt")  # Ganti path dengan model yang sesuai

@app.post("/detect")
async def detect_objects(image_file: UploadFile = File(...)):
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))

    results = model_driver.predict(image)

    all_box_list = []
    all_conf_list = []
    all_cls_list = []

    for result in results:
        boxes = result.boxes
        box_list = []
        conf_list = []
        cls_list = []

        for box in boxes:
            conf = round(float(box.conf), 2)
            cls = int(box.cls)

            if conf >= 0.25:  # Pengendara dengan class ID 0
                box_data = [int(x) for x in box.xyxy[0].tolist()]
                box_list.append(box_data)
                conf_list.append(conf)
                cls_list.append(cls)

        all_box_list.append(box_list)
        all_conf_list.append(conf_list)
        all_cls_list.append(cls_list)

    data = {
        'image': [image_file.filename] * len(all_box_list),
        'boxes': all_box_list,
        'confidence': all_conf_list,
        'classes': all_cls_list
    }

    df = pd.DataFrame(data)

    # Function to crop images
    def crop_image(row):
        img = np.array(image)
        pred_box = row['boxes'][0]  # Get the first bounding box
        cropped_img = img[pred_box[1]:pred_box[3], pred_box[0]:pred_box[2]]
        return Image.fromarray(cropped_img)

    # Create a new column 'cropped_image' with the cropped image results in PIL.Image format
    df['cropped_image'] = df.apply(crop_image, axis=1)

    # Save the cropped image results into BytesIO and return it as a response
    output = io.BytesIO()
    df['cropped_image'].iloc[0].save(output, format='JPEG')

    # Use YOLOv8 to predict on the cropped images
    results = model_object.predict(Image.open(output))
    results=results[0]

    # Initialize an empty image to draw boxes
    img = Image.open(output)
    draw = ImageDraw.Draw(img)

    # Process results and draw bounding boxes on the image
    for result in results.boxes:
        # Extract box coordinates and class name
        x1, y1, x2, y2 = [round(x) for x in result.xyxy[0].tolist()]
        class_name = result.cls[0].item()

        # Draw the bounding box on the image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 20), f"{class_name}", fill="red")

    # Save the image to BytesIO
    output_image = io.BytesIO()
    img.save(output_image, format="JPEG")
    output_image.seek(0)

    return StreamingResponse(content=output_image, media_type="image/jpeg")