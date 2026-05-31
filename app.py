from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import gdown

app = Flask(__name__)

MODEL_PATH = "grape_model.h5"
GDRIVE_ID = "1qqmmBmnHZRYjvOZYKOvIUWHzkrJ2kL8o"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

classes = ["Early", "Healthy", "Moderate", "Severe"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    result = classes[class_index]
    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2)
    )

if __name__ == '__main__':
    app.run(debug=True)
