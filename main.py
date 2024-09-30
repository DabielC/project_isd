import base64
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import torchvision.transforms.v2 as transforms
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

def face_crop(image):
    model = YOLO('https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8l-face.pt')
    
    desired_size = (190, 250)
    padding_factor = 0.1

    results = model.predict(image, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()

    if boxes:
        box = boxes[0]

        # Calculate padding
        width = box[2] - box[0]
        height = box[3] - box[1]
        pad_w = width * padding_factor
        pad_h = height * padding_factor

        # Adjust the bounding box with padding
        x1 = max(0, int(box[0] - pad_w))
        y1 = max(0, int(box[1] - pad_h))
        x2 = min(image.width, int(box[2] + pad_w))
        y2 = min(image.height, int(box[3] + pad_h))

        # Crop the object from the image
        crop_obj = image.crop((x1, y1, x2, y2))

        # Resize the cropped object to the desired size
        crop_obj_resized = crop_obj.resize(desired_size)
        
        return crop_obj_resized
    else:
        print("No objects detected")
    
    return None
 

def base64_to_image(base64_string):
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return image

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((190, 250)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return transform(image)
    

def predict_ResNet(image_tensor):
    # Initialize the model
    model = models.resnet50()
    model.fc = nn.Linear(2048, 5)
    model.load_state_dict(torch.load("model_2.pt", map_location=torch.device('cpu')), strict=False)
    model.eval()

    # Initialize and fit the LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(['Oblong', 'Round', 'Oval', 'Heart', 'Square'])
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0)
        outputs = model(image_tensor)

        confidences = F.softmax(outputs, dim=1)
        predicted_class_index = torch.argmax(confidences, dim=1).item()
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
        confidence_scores = confidences.squeeze().tolist()
        confidence_mapping = {label: score for label, score in zip(label_encoder.classes_, confidence_scores)}
        
    return predicted_class_label, confidence_mapping

def predict_MobileNet(image_tensor):
    # Initialize the model
    model = models.mobilenet_v3_large()
    num_features = model.classifier[3].in_features
    model.fc = nn.Linear(num_features, 5)
    model.load_state_dict(torch.load("https://drive.google.com/file/d/1FpMmRSeLrUGitUD-bBIgMt6qDB4Sfkaw/view?usp=drive_link", map_location=torch.device('cpu')), strict=False)
    model.eval()

    # Initialize and fit the LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(['Oblong', 'Round', 'Oval', 'Heart', 'Square'])
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0)
        outputs = model(image_tensor)
        
        confidences = F.softmax(outputs, dim=1)
        predicted_class_index = torch.argmax(confidences, dim=1).item()
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
        confidence_scores = confidences.squeeze().tolist()
        confidence_mapping = {label: score for label, score in zip(label_encoder.classes_, confidence_scores)}
        
    return predicted_class_label, confidence_mapping

def predict_YOLO(image):
    model = YOLO('https://drive.google.com/file/d/13IEnHw4mKOlkj9CBq7uxKXNmaIozBUwp/view?usp=drive_link')
    results = model(image)

    predicted_class_index = results[0].probs.top1
    predicted_class_name = results[0].names[predicted_class_index]

    dict = {}
    for i, prob in enumerate(results[0].probs.data.tolist()):
        dict[results[0].names[i]] = prob
    return predicted_class_name, dict

class get_info(BaseModel):
    model : str
    image: str

app = FastAPI()

@app.post("/predict/")
async def predicted(info: get_info):
    if info.model == "test_face_crop":
        cropped_face = face_crop(base64_to_image(info.image))

        if cropped_face:
            # Convert PIL Image to numpy array
            cropped_face = np.array(cropped_face)
            
            # Reshape the array to (3, 190, 250)
            cropped_face = np.transpose(cropped_face, (2, 0, 1))
            
            # Create a figure and axis
            fig, ax = plt.subplots()
            
            # Display the image
            # We need to transpose back for correct display
            ax.imshow(np.transpose(cropped_face, (1, 2, 0)))
            
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set a title
            ax.set_title("Cropped Face")
            plt.show()
    elif info.model[:5] == "other":
        image_tensor = transform_image(face_crop(base64_to_image(info.image)))
        print(image_tensor.shape)
        if info.model == "other:mobilenet":
            predicted_class, class_confidences = predict_MobileNet(image_tensor)
            
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence scores: {class_confidences}")
            
            return {
                "predicted_class": predicted_class,
                "confidence_scores": class_confidences
            }
    elif info.model == "yolov8":
        predicted_class = predict_YOLO(face_crop(base64_to_image(info.image)))
        return {
            "predicted_class": predicted_class[0],
            "confidence_scores": predicted_class[1]
        }

