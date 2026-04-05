import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Define your class names
class_names = ['benign', 'malignant', 'normal']

# Function to evaluate model and return detailed DataFrame
def evaluate_model(model_path, model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Classification report as a dictionary
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Confusion matrix for per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()

    # Add per-class accuracy values
    # For macro/weighted avg rows, we'll leave NaN for accuracy
    accuracy_col = list(per_class_acc) + [np.nan, np.nan, np.mean(per_class_acc)]
    df['accuracy'] = accuracy_col

    # Round for neatness
    df = df.round(3)

    return df

# Example usage
# You must define or import your model architectures exactly as trained:
# Example:
# model_resnet101 = torchvision.models.resnet101(pretrained=False)
# model_resnet101.fc = nn.Linear(model_resnet101.fc.in_features, 3)
# model_vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=3)

model_resnet101 = ...  # define your ResNet-101
model_vit = ...        # define your ViT

# Evaluate both models
df_resnet = evaluate_model("best_resnet101.pth", model_resnet101, test_loader)
df_vit = evaluate_model("best_vit.pth", model_vit, test_loader)

print("\n📊 Performance Metrics for ResNet-101:\n")
print(df_resnet)

print("\n📊 Performance Metrics for Vision Transformer (ViT):\n")
print(df_vit)
