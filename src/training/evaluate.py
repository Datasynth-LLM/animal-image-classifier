import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models.cnn import CustomCNN
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/confusion_matrix.png")
print("\n✅ Confusion matrix saved to: reports/confusion_matrix.png")
