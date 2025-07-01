import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "skin_disease_model.pth"
classes_path = "classes.txt"


def load_classes():
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def initialize_model(num_classes):
    try:
        model = models.densenet121(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        return model, 'densenet121'
    except:
        print("Trying ResNet152 architecture...")
        model = models.resnet152(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.32),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.24),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.16),
            nn.Linear(128, num_classes)
        )
        return model, 'resnet152'


transform_densenet = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = None
model_type = None
transform = None

try:
    classes = load_classes()
    num_classes = len(classes)
    model, model_type = initialize_model(num_classes)
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            print(f"Model loaded successfully: {model_type} with {num_classes} classes")
            print(f"Using image size: {'512x512' if model_type == 'densenet121' else '224x224'}")
            transform = transform_densenet if model_type == 'densenet121' else transform_resnet
        except Exception as load_error:
            print(f"Error loading state dict, trying alternative approach: {load_error}")
            try:
                model = torch.load(model_path, map_location=device)
                model.eval()
                model_type = 'complete_model'
                transform = transform_resnet
                print(f"Loaded complete model with {num_classes} classes")
            except Exception as e2:
                print(f"Failed to load model: {e2}")
                model = None
    else:
        print(f"Warning: Model file {model_path} not found")
        model = None
except Exception as e:
    print(f"Error during model initialization: {e}")
    model = None
    classes = []

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        current_transform = transform if transform is not None else transform_resnet
        input_tensor = current_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        print(f"Image shape: {input_tensor.shape}")
        print(f"Top 5 predictions:")
        
        results = []
        for i in range(5):
            idx = top5_idx[0][i].item()
            prob = top5_prob[0][i].item()
            print(f"  {i+1}. {classes[idx]}: {prob*100:.2f}%")
            results.append({
                'condition': classes[idx],
                'confidence': float(prob * 100)
            })
        
        return jsonify({
            'success': True,
            'predictions': results,
            'topPrediction': results[0]
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': model_type,
        'num_classes': len(classes),
        'device': str(device)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)