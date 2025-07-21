import os
import json
import matplotlib.pyplot as plt
import torch

# Load training history
with open("reports/training_history.json", "r") as f:
    history = json.load(f)

# Plot Accuracy vs Epoch
plt.figure(figsize=(8, 5))
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.savefig("reports/accuracy_vs_epoch.png")
print("📈 accuracy_vs_epoch.png saved.")

# Plot Loss vs Epoch
plt.figure(figsize=(8, 5))
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.savefig("reports/loss_vs_epoch.png")
print("📉 loss_vs_epoch.png saved.")

# Save classification report (generated below)
from sklearn.metrics import classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models.cnn import CustomCNN

DATA_DIR = "data/processed"
MODEL_PATH = "models/cnn_baseline/best_model.pth"
NUM_CLASSES = 15
CLASS_NAMES = [
    "Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", "Elephant",
    "Giraffe", "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.cpu().tolist())

report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
with open("reports/classification_report.txt", "w") as f:
    f.write(report_text)

print("✅ classification_report.txt saved at reports/classification_report.txt")

# ✅ Extra: Run evaluate.py to generate confusion matrix
import subprocess
print("📊 Running evaluate.py to generate confusion matrix...")
subprocess.run(["python", "scripts/evaluate.py"])
