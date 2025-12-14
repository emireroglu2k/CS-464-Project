if __name__ == "__main__":
    import os
    import torch
    import json
    import joblib
    import numpy as np
    from PIL import Image
    from torchvision import models, transforms
    from custom_knn import CustomKNN
    import random

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the config settings
    with open("model_config.json", "r") as f:
        config = json.load(f)
    
    USE_RESNET = config["use_resnet"]
    INPUT_SIZE = tuple(config["input_size"])

    print(f"Loading model... (Using ResNet: {USE_RESNET})")
    knn = joblib.load("custom_knn_model.pkl")

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    # setup resnet if needed
    feature_extractor = None
    if USE_RESNET:
        resnet = models.resnet50(pretrained=True).to(DEVICE)
        resnet.eval()
        feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

    # setup transformation according to resnet or raw pixels
    if USE_RESNET:
        norm_mean = [0.485,0.456,0.406]
        norm_std  = [0.229,0.224,0.225]
    else:
        norm_mean = [0.5, 0.5, 0.5]
        norm_std  = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    def predict_image(path):
        img = Image.open(path).convert("RGB")
        t = transform(img).unsqueeze(0).to(DEVICE)

        # convert to numbers (features or raw pixels)
        if USE_RESNET:
            with torch.no_grad():
                feat = feature_extractor(t)
                feat = feat.view(feat.size(0), -1).cpu().numpy()
        else:
            feat = t.view(t.size(0), -1).cpu().numpy()

        # predict using our custom knn algorithm
        pred_label_idx = knn.predict(feat)[0]
        breed = class_names[pred_label_idx]
        
        print(f"\nImage: {path}")
        print(f"Predicted breed: {breed}")
        return breed

    # pick one random image to test
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

    # pick a random breed folder
    random_breed = random.choice(os.listdir(TEST_DIR))
    breed_folder = os.path.join(TEST_DIR, random_breed)

    # pick a random image inside that folder
    random_image = random.choice(os.listdir(breed_folder))
    img_path = os.path.join(breed_folder, random_image)

    print(f"Testing on a random image from: {random_breed}")
    predict_image(img_path)