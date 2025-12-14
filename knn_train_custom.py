if __name__ == "__main__":
    import os
    import torch
    from torchvision import models, transforms, datasets
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score
    import numpy as np
    from tqdm import tqdm
    import joblib
    import json
    from custom_knn import CustomKNN 

    # false => raw pixels mode
    # true => resnet mode (high accuracy)
    USE_RESNET = True

    K_VALUES_TO_TRY = [10, 25, 50, 100] 

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # UPDATE THESE PATHS IF NEEDED
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "preprocessed_train")
    VAL_DIR   = os.path.join(BASE_DIR, "dataset", "validation")

    BATCH_SIZE = 32

    # --- SETUP IMAGES ---
    if USE_RESNET:
        input_size = (224, 224)
        # Standard normalization for pre-trained models
        norm_mean = [0.485,0.456,0.406]
        norm_std  = [0.229,0.224,0.225]
    else:
        # for raw pixels use smaller input
        input_size = (64, 64) 
        norm_mean = [0.5, 0.5, 0.5]
        norm_std  = [0.5, 0.5, 0.5]

    data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Load the folders
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transform)
    val_dataset   = datasets.ImageFolder(VAL_DIR, transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- FEATURE EXTRACTOR ---
    feature_extractor = None
    if USE_RESNET:
        print("Model: Using ResNet50")
        resnet = models.resnet50(pretrained=True).to(DEVICE)
        resnet.eval()
        # Remove the last layer so we get features, not predictions
        feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
    else:
        print("Model: Using Raw Pixels")

    # This helper function reads images from the folder and converts them to numbers
    def get_data(loader):
        all_data = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Loading Data"):
                imgs = imgs.to(DEVICE)
                
                if USE_RESNET:
                    # Pass image through ResNet to get "smart" features
                    feats = feature_extractor(imgs)
                    feats = feats.view(feats.size(0), -1) 
                    all_data.append(feats.cpu().numpy())
                else:
                    # Just flatten the image into a long list of pixel colors
                    flat_imgs = imgs.view(imgs.size(0), -1)
                    all_data.append(flat_imgs.cpu().numpy())
                
                all_labels.append(labels.numpy())

        # Stack them into one big numpy array
        return np.vstack(all_data), np.hstack(all_labels)

    # get the data
    print("Preparing Training Data...")
    train_X, train_y = get_data(train_loader)

    print("Preparing Validation Data...")
    val_X, val_y = get_data(val_loader)

    # train with different K values
    best_acc = 0
    best_k = 0
    best_model = None

    for k in K_VALUES_TO_TRY:
        print(f"\n--- Testing Custom KNN with K={k} ---")
        
        # create classifier
        knn = CustomKNN(k=k)
        knn.fit(train_X, train_y) 
        
        print("Predicting...")
        val_preds = knn.predict(val_X)
        
        acc = accuracy_score(val_y, val_preds)
        print(f"Validation Accuracy (K={k}): {acc:.4f}")
        
        # check if the current one is the best one
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_model = knn

    print(f"\nWINNER: Best K was {best_k} with Accuracy: {best_acc}")

    # save the results
    joblib.dump(best_model, "custom_knn_model.pkl")
    
    # save a config file so the test script knows what is used
    config = {"use_resnet": USE_RESNET, "input_size": input_size}
    with open("model_config.json", "w") as f:
        json.dump(config, f)

    with open("class_names.json", "w") as f:
        json.dump(train_dataset.classes, f)
        
    print("Saved best model and config.")