# ======================================================
# OFAT Hyperparameter Study with Detailed Evaluation
# ResNet-18 FROM SCRATCH
# ======================================================

import os
import csv
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ------------------------------------------------------
# TRANSFORMS
# ------------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ------------------------------------------------------
# RESNET-18 FROM SCRATCH
# ------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.in_c = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make(block, 64, layers[0], 1)
        self.layer2 = self._make(block, 128, layers[1], 2)
        self.layer3 = self._make(block, 256, layers[2], 2)
        self.layer4 = self._make(block, 512, layers[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make(self, block, out_c, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_c, out_c, s))
            self.in_c = out_c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# ------------------------------------------------------
# TRAIN / EVAL UTILITIES
# ------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        preds = model(x).argmax(1).cpu().numpy()
        y_true.extend(y.numpy())
        y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)

# ------------------------------------------------------
# MAIN EXPERIMENT
# ------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.freeze_support()

    base_dir = r"C:\Users\user\Desktop\Homeworks\CS464\ml\dataset"
    out_dir = "ofat_detailed_results"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = datasets.ImageFolder(os.path.join(base_dir, "train"), train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(base_dir, "validation"), eval_transform)
    test_ds  = datasets.ImageFolder(os.path.join(base_dir, "test"), eval_transform)

    class_names = train_ds.classes

    baseline = dict(lr=3e-4, batch_size=32, weight_decay=1e-4)

    experiments = {
        "learning_rate": [1e-2, 1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 1e-6],
        "batch_size": [16, 32, 64],
        "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
    }

    epochs = 10

    summary_csv = open(os.path.join(out_dir, "summary.csv"), "w", newline="")
    summary_writer = csv.writer(summary_csv)
    summary_writer.writerow([
        "parameter", "value",
        "best_val_accuracy",
        "test_accuracy_macro_f1"
    ])

    for param, values in experiments.items():
        for value in values:

            lr, bs, wd = baseline["lr"], baseline["batch_size"], baseline["weight_decay"]
            if param == "learning_rate": lr = value
            if param == "batch_size": bs = value
            if param == "weight_decay": wd = value

            tag = f"{param}_{value}"
            print(f"\n=== Testing {tag} ===")

            train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=4)
            val_loader   = DataLoader(val_ds, bs, shuffle=False, num_workers=4)
            test_loader  = DataLoader(test_ds, bs, shuffle=False, num_workers=4)

            model = ResNet18(len(class_names)).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.CrossEntropyLoss()

            best_val = 0.0
            for _ in tqdm(range(epochs)):
                train_epoch(model, train_loader, optimizer, criterion, device)
                y_val_true, y_val_pred = get_predictions(model, val_loader, device)
                val_acc = (y_val_true == y_val_pred).mean()
                best_val = max(best_val, val_acc)

            y_test_true, y_test_pred = get_predictions(model, test_loader, device)

            cm = confusion_matrix(y_test_true, y_test_pred)
            report = classification_report(
                y_test_true, y_test_pred,
                target_names=class_names, output_dict=True
            )

            # save confusion matrix
            np.savetxt(
                os.path.join(out_dir, f"{tag}_confusion_matrix.csv"),
                cm, delimiter=",", fmt="%d"
            )

            # save classification report
            with open(os.path.join(out_dir, f"{tag}_classification_report.txt"), "w") as f:
                f.write(classification_report(
                    y_test_true, y_test_pred,
                    target_names=class_names
                ))

            summary_writer.writerow([
                param, value,
                round(best_val, 4),
                round(report["macro avg"]["f1-score"], 4)
            ])

    summary_csv.close()
    print("\nAll detailed results saved to:", out_dir)
