import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121  # Switched to DenseNet121
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# --- Data Preparation Functions ---
def parse_info_file(info_path):
    column_names = ["reference", "background_tissue", "abnormality_class", "severity", "x_coord", "y_coord", "radius"]
    data = []
    with open(info_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 7:
                parts = parts[:7]
            elif len(parts) < 7:
                parts.extend([None] * (7 - len(parts)))
            data.append(parts)
    df = pd.DataFrame(data, columns=column_names)
    df['x_coord'] = pd.to_numeric(df['x_coord'], errors='coerce')
    df['y_coord'] = pd.to_numeric(df['y_coord'], errors='coerce')
    df['radius'] = pd.to_numeric(df['radius'], errors='coerce')
    return df

def filter_metadata(metadata, image_dir):
    metadata = metadata.dropna(subset=['x_coord', 'y_coord', 'radius'])
    metadata = metadata[metadata['radius'] > 0]
    available_references = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.pgm')]
    metadata = metadata[metadata['reference'].isin(available_references)]
    return metadata.reset_index(drop=True)

def load_images(image_dir, metadata):
    images = []
    for ref in metadata['reference']:
        path = os.path.join(image_dir, f"{ref}.pgm")
        if os.path.exists(path):
            images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    return np.array(images)

def extract_roi(image, x, y, radius, target_size=(128, 128)):
    x, y, radius = int(x), int(y), int(radius)
    x_min, x_max = max(0, x - radius), min(image.shape[1], x + radius)
    y_min, y_max = max(0, y - radius), min(image.shape[0], y + radius)
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None
    return cv2.resize(roi, target_size)

def prepare_rois(images, metadata, target_size=(128, 128)):
    rois, labels = [], []
    for i, row in metadata.iterrows():
        roi = extract_roi(images[i], row['x_coord'], row['y_coord'], row['radius'], target_size)
        if roi is not None:
            rois.append(roi)
            labels.append(1 if row['severity'] == 'M' else 0)
    return np.array(rois), np.array(labels)

# --- Focal Loss Function ---
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fixed

# --- Generator Function for Explicit Casting and Class Weights ---
def generator_with_class_weights(data_gen, X_data, y_data, class_weights):
    for x_batch, y_batch in data_gen.flow(X_data, y_data, batch_size=32):  # Increased batch size
        # Calculate sample weights based on class weights
        weight_batch = np.array([class_weights[np.argmax(label)] for label in y_batch])
        yield x_batch.astype('float32'), y_batch.astype('float32')

# --- Model Building Function (DenseNet121) ---
def build_densenet_model(input_shape=(128, 128, 3), num_classes=2):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

    # Unfreeze more layers (e.g., last 30 layers)
    for layer in base_model.layers[-30:]:  # Unfreeze more layers
        layer.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=focal_loss(), metrics=['accuracy'])
    return model

# --- Main Pipeline ---
if __name__ == "__main__": 
    info_path = "C:\\Users\\mery\\OneDrive\\Desktop\\New folder\\INT.Jena\\Info.txt"
    image_directory = "C:\\Users\\mery\\OneDrive\\Desktop\\New folder\\INT.Jena\\all-mias"

    metadata = parse_info_file(info_path)
    metadata = filter_metadata(metadata, image_directory)
    images = load_images(image_directory, metadata)
    rois, roi_labels = prepare_rois(images, metadata)

    X_train, X_val, y_train, y_val = train_test_split(rois, roi_labels, test_size=0.2, random_state=42)
    X_train = (X_train / 255.0).astype('float32')
    X_val = (X_val / 255.0).astype('float32')
    X_train_rgb = np.repeat(X_train[..., np.newaxis], 3, axis=-1)
    X_val_rgb = np.repeat(X_val[..., np.newaxis], 3, axis=-1)
    y_train = to_categorical(y_train, num_classes=2).astype('float32')
    y_val = to_categorical(y_val, num_classes=2).astype('float32')

    class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
    class_weights_dict = dict(enumerate(class_weights))

    datagen = ImageDataGenerator(rotation_range=50, width_shift_range=0.3, height_shift_range=0.3, shear_range=0.3,
                                 zoom_range=0.3, horizontal_flip=True, fill_mode='nearest', brightness_range=[0.5, 1.5])
    datagen.fit(X_train_rgb)

    train_gen = generator_with_class_weights(datagen, X_train_rgb, y_train, class_weights_dict)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

    model = build_densenet_model(input_shape=(128, 128, 3), num_classes=2)
    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train_rgb) // 32,  # Increased batch size to 32
        validation_data=(X_val_rgb, y_val),
        epochs=100,  # Increase epochs to 100
        callbacks=[early_stopping, lr_scheduler],
        class_weight=None  # Class weights handled by generator
    )

    results = model.evaluate(X_val_rgb, y_val)
    print(f"Validation Accuracy: {results[1] * 100:.2f}%")
