from fastapi import FastAPI, UploadFile, File
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import numpy as np
import io
from fastapi.responses import StreamingResponse

app = FastAPI()
model = YOLO("C:/Users/ziyad/Documents/Bangkit/deployment/best.pt")

@app.post("/detect")
async def detect_objects(image_file: UploadFile = File(...)):
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))

    results = model.predict(image)

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

    # Fungsi untuk melakukan crop gambar
    def crop_image(row):
        img = np.array(image)
        pred_box = row['boxes'][0]  # Mengambil kotak pembatas pertama
        cropped_img = img[pred_box[1]:pred_box[3], pred_box[0]:pred_box[2]]
        return Image.fromarray(cropped_img)

    # Buat kolom baru 'cropped_image' dengan hasil crop gambar dalam bentuk PIL.Image
    df['cropped_image'] = df.apply(crop_image, axis=1)

    # Simpan hasil crop gambar ke dalam BytesIO dan kembalikan sebagai response
    output = io.BytesIO()
    df['cropped_image'].iloc[0].save(output, format='JPEG')
    output.seek(0)
    
    return StreamingResponse(content=output, media_type="image/jpeg")
