import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data_preprocessing import parse_info_file, filter_metadata, load_images, prepare_rois

# Argument parser to select model
parser = argparse.ArgumentParser(description="Evaluate a trained model")
parser.add_argument("--model", type=str, choices=["vgg16", "densenet", "custom_cnn"], required=True, help="Choose model to evaluate")
args = parser.parse_args()

# Load Data
info_path = "data/all-mias/Info.txt"
image_directory = "data/all-mias"

metadata = parse_info_file(info_path)
metadata = filter_metadata(metadata, image_directory)
images = load_images(image_directory, metadata)
rois, roi_labels = prepare_rois(images, metadata)

# Normalize & Convert to RGB
X_val = np.repeat((rois / 255.0)[..., np.newaxis], 3, axis=-1).astype('float32')
y_val = np.array(roi_labels)

# Load Model
model_path = f"breast_cancer_{args.model}.h5"
model = load_model(model_path)

# Evaluate Model
y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=["Benign", "Malignant"]))

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {args.model}")
plt.show()

# Print Accuracy
accuracy = np.sum(y_pred == y_val) / len(y_val)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
