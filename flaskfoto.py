from flask import Flask, request, send_file
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import numpy as np
import io

app = Flask(__name__)
model_driver = YOLO("C:/Users/ziyad/Documents/Bangkit/deployment/best.pt")
model_object = YOLO("C:/Users/ziyad/Documents/Bangkit/deployment/best2.pt")  # Ganti path dengan model yang sesuai

@app.route('/detect', methods=['POST'])
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
    cropped_images = []

    for result in results:
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

    data = {
        'image': [file.filename] * len(all_box_list),
        'boxes': all_box_list,
        'confidence': all_conf_list,
        'classes': all_cls_list
    }

    df = pd.DataFrame(data)

    def crop_image(row):
        img = Image.open(image_path)
        pred_box = row['boxes'][0]
        cropped_img = img.crop(pred_box)  # Crop the image using box coordinates
        return cropped_img  # Return the Image object

    df['cropped_image'] = df.apply(crop_image, axis=1)

    output_buffer = io.BytesIO()
    df['cropped_image'].iloc[0].save(output_buffer, format='PNG')
    output_buffer.seek(0)

    # Return the encoded image as a PNG file
    return send_file(output_buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
