from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from utils.focal_loss import focal_loss  # Import focal loss

def build_densenet_model(input_shape=(128, 128, 3), num_classes=2):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
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
