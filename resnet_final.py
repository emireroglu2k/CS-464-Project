# ======================================================
# Final Training Run with Detailed Evaluation
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
        strides = [stride] + [1] * (blocks - 1)
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
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(1).eq(y).sum().item()
        total += y.size(0)
    return correct / total


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
# MAIN
# ------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.freeze_support()

    base_dir = r"C:\Users\user\Desktop\Homeworks\CS464\ml\dataset" 
    out_dir = "final_run_results"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = datasets.ImageFolder(os.path.join(base_dir, "train"), train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(base_dir, "validation"), eval_transform)
    test_ds  = datasets.ImageFolder(os.path.join(base_dir, "test"), eval_transform)

    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)

    model = ResNet18(len(class_names)).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    # ---------------- TRAINING ----------------
    epochs = 50
    for epoch in range(1, epochs + 1):
        train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch:02d}/{epochs}] | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # ---------------- FINAL TEST EVALUATION ----------------
    y_true, y_pred = get_predictions(model, test_loader, device)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )

    # save confusion matrix
    np.savetxt(
        os.path.join(out_dir, "confusion_matrix.csv"),
        cm, delimiter=",", fmt="%d"
    )

    # save classification report
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # save summary
    macro_f1 = classification_report(
        y_true, y_pred,
        output_dict=True,
        zero_division=0
    )["macro avg"]["f1-score"]

    with open(os.path.join(out_dir, "summary2.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["learning_rate", "batch_size", "weight_decay", "macro_f1"])
        writer.writerow([3e-4, 16, 1e-6, round(macro_f1, 4)])

    print("\nFinal evaluation saved to:", out_dir)
