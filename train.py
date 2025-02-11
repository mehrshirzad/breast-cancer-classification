import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from utils.data_preprocessing import parse_info_file, filter_metadata, load_images, prepare_rois
from models.vgg16_model import build_vgg16_model
from models.densenet_model import build_densenet_model
from models.custom_cnn import build_custom_cnn

# Argument parser to choose model
parser = argparse.ArgumentParser(description="Train a model for breast cancer classification")
parser.add_argument("--model", type=str, choices=["vgg16", "densenet", "custom_cnn"], required=True, help="Choose model to train")
args = parser.parse_args()

# Load Data
info_path = "data/all-mias/Info.txt"
image_directory = "data/all-mias"

metadata = parse_info_file(info_path)
metadata = filter_metadata(metadata, image_directory)
images = load_images(image_directory, metadata)
rois, roi_labels = prepare_rois(images, metadata)

X_train, X_val, y_train, y_val = train_test_split(rois, roi_labels, test_size=0.2, random_state=42)

X_train = np.repeat((X_train / 255.0)[..., np.newaxis], 3, axis=-1).astype('float32')
X_val = np.repeat((X_val / 255.0)[..., np.newaxis], 3, axis=-1).astype('float32')
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

# Select model
if args.model == "vgg16":
    model = build_vgg16_model(input_shape=(128, 128, 3), num_classes=2)
elif args.model == "densenet":
    model = build_densenet_model(input_shape=(128, 128, 3), num_classes=2)
elif args.model == "custom_cnn":
    model = build_custom_cnn(input_shape=(128, 128, 3), num_classes=2)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16)

# Save the model
model.save(f"breast_cancer_{args.model}.h5")
