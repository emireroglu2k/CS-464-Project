import torch
import json
import joblib
import numpy as np
from PIL import Image
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load saved classifier + classes
# ------------------------------
knn = joblib.load("knn_breed_classifier.pkl")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ------------------------------
# Load pretrained ResNet50 feature extractor
# ------------------------------
resnet = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove FC
feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()

# ------------------------------
# Image transform (same as training)
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ------------------------------
# Function: predict a single image
# ------------------------------
def predict_image(path):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = feature_extractor(t)
        feat = feat.view(feat.size(0), -1)  # flatten
        feat = feat.cpu().numpy()

    pred_label = knn.predict(feat)[0]
    pred_conf  = knn.kneighbors(feat, return_distance=True)[0]  # distances

    print("\nImage:", path)
    print("Predicted breed:", class_names[pred_label])
    print("Nearest-neighbor distances:", pred_conf)

    return class_names[pred_label]

# ------------------------------
# Try an image
# ------------------------------
if __name__ == "__main__":
    img_path = r"C:\Users\user\Downloads\scot.jpg"
    predict_image(img_path)
