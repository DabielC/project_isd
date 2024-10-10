from PIL import Image
from io import BytesIO

import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms.v2 as transforms
from sklearn.preprocessing import LabelEncoder
from ultralytics import YOLO

def face_crop(image):
    model = YOLO('model/face_crop/yolov8l-face (1).pt')
    
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

def predict_MobileNet(image_tensor):
    # Initialize the model
    model = models.mobilenet_v3_large()
    num_features = model.classifier[3].in_features
    model.fc = nn.Linear(num_features, 5)
    model.load_state_dict(torch.load("model/mobilenet_casia_web_face_augmentation/model_MobileNetV3_Greyscal_Augment.pt", map_location=torch.device('cpu')), strict=False)
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
        
    return {"class" : predicted_class_label, "score" : confidence_mapping}

def predict_YOLO(image):
    model = YOLO('model/yolov8_imagenet/trained_yolov8x-cls_2 (1).pt')
    results = model(image)

    predicted_class_index = results[0].probs.top1
    predicted_class_name = results[0].names[predicted_class_index]

    dict = {}
    for i, prob in enumerate(results[0].probs.data.tolist()):
        dict[results[0].names[i]] = prob
    return {"class" : predicted_class_name, "score" : dict}

def vote(mobile_pred, yolo_pred):
    mobile_conf = mobile_pred["score"]
    yolo_conf = yolo_pred["score"]
    average_score = {
        'Heart': (yolo_conf['Heart'] + mobile_conf['Heart']) / 2,
        'Oblong': (yolo_conf['Oblong'] + mobile_conf['Oblong']) / 2,
        'Oval': (yolo_conf['Oval'] + mobile_conf['Oval']) / 2,
        'Round': (yolo_conf['Round'] + mobile_conf['Round']) / 2,
        'Square': (yolo_conf['Square'] + mobile_conf['Square']) / 2
    }

    voted_class = max(average_score, key=average_score.get)
    
    return {"class" : voted_class, "score": average_score}