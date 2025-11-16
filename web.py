import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms


app = Flask(__name__)

class_names = [
    'Speed Limit 20 km/h',
    'Speed Limit 30 km/h',
    'Speed Limit 50 km/h',
    'Speed Limit 60 km/h',
    'Speed Limit 70 km/h',
    'Speed Limit 80 km/h',
    'End of Speed Limit 80 km/h',
    'Speed Limit 100 km/h',
    'Speed Limit 120 km/h',
    'No passing',
    'No passing for vechiles over 3.5 metric tons',
    'Right-of-way at the next intersection',
    'Priority road',
    'Yield',
    'Stop',
    'No vechiles'
]

class MyModel(nn.Module):
    def __init__(self, num_classes=16):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 60, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(60, 60, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(60, 30, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(30, 30, kernel_size=3, padding=1)
        self.drop = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(30*8*8, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.pool(x)
        x = self.drop(x)
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


MODEL_PATH = 'model1.pth'  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel(num_classes=16).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # channel
    img = np.expand_dims(img, axis=0)  # batch
    img_tensor = torch.FloatTensor(img).to(device)
    return img_tensor


def model_predict(img_path):
    img_tensor = preprocess_image(img_path)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    f = request.files['file']
    if f.filename == '':
        return "No selected file"
    basepath = os.path.dirname(__file__)
    upload_dir = os.path.join(basepath, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, secure_filename(f.filename))
    f.save(file_path)
    preds = model_predict(file_path)
    return preds


if __name__ == '__main__':
    app.run(port=5001, debug=True)
