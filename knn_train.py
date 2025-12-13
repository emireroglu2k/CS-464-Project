if __name__ == "__main__":
    import os
    import torch
    from torchvision import models, transforms, datasets
    from torch.utils.data import DataLoader
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np
    from tqdm import tqdm
    import joblib
    import json

    # ------------------------------
    # Paths & settings
    # ------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_DIR = r"C:\Users\user\Desktop\ml\dataset\preprocessed_train"
    VAL_DIR   = r"C:\Users\user\Desktop\ml\dataset\validation"
    TEST_DIR  = r"C:\Users\user\Desktop\ml\dataset\test"

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    K = 8  # number of neighbors for KNN

    # ------------------------------
    # Transforms
    # ------------------------------
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # ------------------------------
    # Load Datasets
    # ------------------------------
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=eval_transform)
    val_dataset   = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
    test_dataset  = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("Classes:", train_dataset.classes)
    print("Number of breeds:", len(train_dataset.classes))
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    # ------------------------------
    # Load pretrained ResNet50 (feature extractor)
    # ------------------------------
    resnet = models.resnet50(pretrained=True)
    resnet = resnet.to(DEVICE)
    resnet.eval()

    # Remove final classification layer
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # [B, 2048, 1, 1]

    # ------------------------------
    # Function: extract features from loader
    # ------------------------------
    def extract_features(loader):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in tqdm(loader):
                imgs = imgs.to(DEVICE)
                feats = feature_extractor(imgs)
                feats = feats.view(feats.size(0), -1)  # flatten [B, 2048]
                all_features.append(feats.cpu().numpy())
                all_labels.append(labels.numpy())

        all_features = np.vstack(all_features)
        all_labels = np.hstack(all_labels)
        return all_features, all_labels

    # ------------------------------
    # Extract features
    # ------------------------------
    print("Extracting train features...")
    train_features, train_labels = extract_features(train_loader)

    print("Extracting validation features...")
    val_features, val_labels = extract_features(val_loader)

    print("Extracting test features...")
    test_features, test_labels = extract_features(test_loader)

    # ------------------------------
    # Train KNN classifier
    # ------------------------------
    knn = KNeighborsClassifier(n_neighbors=K, metric='euclidean', n_jobs=-1)
    knn.fit(train_features, train_labels)

    # ------------------------------
    # Evaluate
    # ------------------------------
    val_preds = knn.predict(val_features)
    test_preds = knn.predict(test_features)

    print("Validation Accuracy:", accuracy_score(val_labels, val_preds))
    print("Validation Classification Report:\n", classification_report(val_labels, val_preds, target_names=val_dataset.classes))

    print("Test Accuracy:", accuracy_score(test_labels, test_preds))
    print("Test Classification Report:\n", classification_report(test_labels, test_preds, target_names=test_dataset.classes))

    print("Validation Confusion Matrix:\n", confusion_matrix(val_labels, val_preds))
    print("Test Confusion Matrix:\n", confusion_matrix(test_labels, test_preds))

    # Save KNN
    joblib.dump(knn, "knn_breed_classifier.pkl")

    # Save class names
    with open("class_names.json", "w") as f:
        json.dump(train_dataset.classes, f)

    print("Model and class names saved.")