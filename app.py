from flask import Flask, request, jsonify, render_template
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import numpy as np
import io
import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = Flask(__name__)

# If you want to start from the pretrained model, load the checkpoint with `VisionEncoderDecoderModel`
processor = TrOCRProcessor.from_pretrained('ziyadazz/OCR-PLAT-NOMOR-INDONESIA')

# TrOCR is a decoder model and should be used within a VisionEncoderDecoderModel
model = VisionEncoderDecoderModel.from_pretrained('ziyadazz/OCR-PLAT-NOMOR-INDONESIA')

model_driver = YOLO("C:/Users/ziyad/Documents/Bangkit/deployment/best.pt")
model_object = YOLO("C:/Users/ziyad/Documents/Bangkit/deployment/best2.pt")  # Ganti path dengan model yang sesuai

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def detect_objects():
    # Get the image file from the request
    file = request.files['image']

    # Save the file temporarily (optional)
    image_path = 'temp_image.jpg'
    file.save(image_path)

    results = model_driver.predict(image_path)

    all_box_list = []
    all_conf_list = []
    all_cls_list = []
    cropped_image_paths = []  # Menyimpan path hasil cropping

    for idx, result in enumerate(results):
        boxes = result.boxes
        box_list = []
        conf_list = []
        cls_list = []

        for box in boxes:
            conf = round(float(box.conf), 2)
            cls = int(box.cls)

            if conf >= 0.25:
                box_data = [int(x) for x in box.xyxy[0].tolist()]
                box_list.append(box_data)
                conf_list.append(conf)
                cls_list.append(cls)

        all_box_list.append(box_list)
        all_conf_list.append(conf_list)
        all_cls_list.append(cls_list)

        # Menyimpan hasil cropping dalam format JPG
        img = Image.open(image_path)
        cropped_img = img.crop(box_list[0])  # Ambil kotak pertama
        cropped_image_path = f'cropped_image_{idx}.jpg'
        cropped_img.save(cropped_image_path)
        cropped_image_paths.append(cropped_image_path)

    data = {
        'image': [file.filename] * len(all_box_list),
        'boxes': all_box_list,
        'confidence': all_conf_list,
        'classes': all_cls_list,
        'cropped_image_paths': cropped_image_paths  # Menambahkan path hasil cropping ke dalam data
    }

    df = pd.DataFrame(data)

    results = model_object.predict(list(df['cropped_image_paths']))

    all_box_list = []
    all_conf_list = []
    all_cls_list = []

    for result in results:
        boxes = result.boxes
        cls_list = []
        box_list = []
        conf_list = []

        for box in boxes:
            conf = round(float(box.conf), 2)
            cls = round(float(box.cls), 2)
            if conf >= 0.5:
                box_data = box.data[0][:4]
                box_data = [int(x) for x in box_data]
                cls_list.append(cls)
                conf_list.append(conf)
                box_list.append(box_data)

        all_box_list.append(box_list)
        all_conf_list.append(conf_list)
        all_cls_list.append(cls_list)

    df["pred_box"] = all_box_list
    df["confidence"] = all_conf_list
    df['cls']=all_cls_list

    rows = []
    for idx, row in df.iterrows():
        image_path = row['cropped_image_paths']
        pred_boxes = row['pred_box']
        confidences = row['confidence']
        classes = row['cls']

        # Loop untuk setiap prediksi dalam satu baris
        for i in range(len(pred_boxes)):
            rows.append({
                "cropped_image_paths": image_path,
                "pred_box": pred_boxes[i],
                "confidence": confidences[i],
                "cls": classes[i]
            })
    new_df = pd.DataFrame(rows)
    def crop_and_save_image(row):
        img = cv2.imread(row['cropped_image_paths'])
        pred_box = row['pred_box']
        cropped_img = img[pred_box[1]:pred_box[3], pred_box[0]:pred_box[2]]

        # Mendapatkan label untuk nama file
        label_mapping = {0: 'exp-date', 1: 'helm', 2: 'licence-plate', 3: 'no-helm'}
        label = label_mapping[row['cls']]

        # Resize semua gambar menjadi 384x384
        cropped_img = cv2.resize(cropped_img, (384, 384), interpolation=cv2.INTER_AREA)

        # Simpan gambar yang sudah dipotong dalam format JPG sesuai dengan label
        cropped_image_path = f'cropped_{label}_{row.name}.jpg'
        cv2.imwrite(cropped_image_path, cropped_img)

        return cropped_image_path
    
    new_df['cropped_image_saved_path'] = new_df.apply(crop_and_save_image, axis=1)
    filtered_df = new_df[new_df['cls'].isin([0.0, 2.0])]

    pred = []

    for imges_path in filtered_df['cropped_image_saved_path']:
        image = Image.open(imges_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pred.append(generated_text)

    filtered_df['Prediksi'] = pred

    filtered_df=filtered_df


    # Return the DataFrame as a JSON response
    return render_template('prediction.html', filtered_df=filtered_df)

if __name__ == '__main__':
    app.run(debug=True)
    
