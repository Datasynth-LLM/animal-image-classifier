import sys
import os

# Add project root to sys.path so imports from src/ work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models.cnn import CustomCNN
import shutil

# --- Configuration ---
DATA_PATH = "data/processed/train"
MODEL_DIR = "models/cnn_baseline"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 15
LEARNING_RATE = 0.001

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Backup Existing Model ---
if os.path.exists(MODEL_PATH):
    backup_path = MODEL_PATH.replace(".pth", "_backup.pth")
    shutil.copy(MODEL_PATH, backup_path)
    print(f"🛡️ Existing best_model.pth backed up to {backup_path}")

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Dataset ---
dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
class_names = dataset.classes

# --- Split into train and val sets ---
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save best model
    if val_accuracy > best_val_accuracy:
        torch.save(model.state_dict(), MODEL_PATH)
        best_val_accuracy = val_accuracy
        print(f"✅ Best model saved with accuracy: {val_accuracy:.4f}")

print("🎉 Training complete.")
