# Cell 1: Import Required Libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, concatenate, UpSampling2D
import os
import cv2
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

print(f"TensorFlow version: {tf.__version__}")

# Cell 2: Define Data Loading and Preprocessing Functions
def load_images_from_folder(folder):
    """Load images from a folder"""
    images = []
    for filename in glob(os.path.join(folder, '*.png')):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def resize_images(images, target_size):
    """Resize images to target size"""
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size)
        resized_images.append(resized_img)
    return np.array(resized_images)

# Set parameters
SIZE = 256
BATCH_SIZE = 8

# Define paths
TRAIN_IMAGES_PATH = 'path_to_training_images'  # Update with your path
TRAIN_MASKS_PATH = 'path_to_training_masks'    # Update with your path
FULL_DATASET_PATH = 'path_to_full_dataset'     # Update with your path

# Cell 3: Load and Preprocess Training Data
# Load training data
train_images = load_images_from_folder(TRAIN_IMAGES_PATH)
train_masks = load_images_from_folder(TRAIN_MASKS_PATH)

# Resize images and masks
train_images = resize_images(train_images, (SIZE, SIZE))
train_masks = resize_images(train_masks, (SIZE, SIZE))

# Normalize images
train_images = train_images / 255.0
train_masks = train_masks / 255.0

# Add channel dimension
train_images = np.expand_dims(train_images, axis=-1)
train_masks = np.expand_dims(train_masks, axis=-1)

# Split into train and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(
    train_images, train_masks, test_size=0.2, random_state=42
)

print(f"Training images shape: {train_images.shape}")
print(f"Training masks shape: {train_masks.shape}")
print(f"Validation images shape: {val_images.shape}")
print(f"Validation masks shape: {val_masks.shape}")

# Cell 4: Define U-Net Model Architecture
def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder
    up6 = Conv2D(512, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv9)
    return model

# Cell 5: Define Custom Metrics
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f))

def jaccard_index(y_true, y_pred):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return intersection / total

# Cell 6: Create and Compile Model
# Create the model
model = unet_model(input_size=(SIZE, SIZE, 1))

# Compile the model with custom metrics
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', dice_coefficient, jaccard_index])

# Print model summary
model.summary()

# Cell 7: Set Up Callbacks and Train the Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define callbacks
checkpoint = ModelCheckpoint('best_model.h5',
                            monitor='val_dice_coefficient',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

early_stopping = EarlyStopping(monitor='val_loss',
                              patience=10,
                              restore_best_weights=True)

# Train the model
history = model.fit(train_images, train_masks,
                    validation_data=(val_images, val_masks),
                    batch_size=BATCH_SIZE,
                    epochs=50,
                    callbacks=[checkpoint, early_stopping])

# Cell 8: Plot Training History
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Plot dice coefficient
    axes[1, 0].plot(history.history['dice_coefficient'], label='Training Dice')
    axes[1, 0].plot(history.history['val_dice_coefficient'], label='Validation Dice')
    axes[1, 0].set_title('Dice Coefficient')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Coefficient')
    axes[1, 0].legend()
    
    # Plot jaccard index
    axes[1, 1].plot(history.history['jaccard_index'], label='Training Jaccard')
    axes[1, 1].plot(history.history['val_jaccard_index'], label='Validation Jaccard')
    axes[1, 1].set_title('Jaccard Index')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Jaccard Index')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Cell 9: Evaluate on Full Dataset in Cycles
# Load the best model
model.load_weights('best_model.h5')

# Load full dataset
full_images = load_images_from_folder(FULL_DATASET_PATH)
full_images = resize_images(full_images, (SIZE, SIZE))
full_images = full_images / 255.0
full_images = np.expand_dims(full_images, axis=-1)

# Process in cycles of 25%
cycle_size = len(full_images) // 4
final_metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

for i in range(4):
    start_idx = i * cycle_size
    end_idx = (i + 1) * cycle_size if i < 3 else len(full_images)
    
    cycle_images = full_images[start_idx:end_idx]
    
    # Generate predictions
    predictions = model.predict(cycle_images)
    binary_predictions = (predictions > 0.5).astype(np.uint8)
    
    # Calculate metrics
    accuracy = accuracy_score(cycle_images.flatten(), binary_predictions.flatten())
    precision = precision_score(cycle_images.flatten(), binary_predictions.flatten())
    recall = recall_score(cycle_images.flatten(), binary_predictions.flatten())
    f1 = f1_score(cycle_images.flatten(), binary_predictions.flatten())
    
    final_metrics['accuracy'].append(accuracy)
    final_metrics['precision'].append(precision)
    final_metrics['recall'].append(recall)
    final_metrics['f1'].append(f1)
    
    print(f"\nCycle {i+1} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Cell 10: Print Final Results and Visualize Predictions
# Print final average metrics
print("\nFinal Average Metrics:")
for metric, values in final_metrics.items():
    print(f"{metric.capitalize()}: {np.mean(values):.4f}")

# Visualize some predictions
fig, axes = plt.subplots(5, 3, figsize=(15, 25))
random_indices = np.random.randint(0, len(full_images), 5)

for i, idx in enumerate(random_indices):
    # Original image
    axes[i, 0].imshow(full_images[idx, :, :, 0], cmap='gray')
    axes[i, 0].set_title('Original Image')
    axes[i, 0].axis('off')
    
    # Prediction
    prediction = model.predict(np.expand_dims(full_images[idx], 0))[0]
    binary_prediction = (prediction > 0.5).astype(np.uint8)
    axes[i, 1].imshow(binary_prediction[:, :, 0], cmap='gray')
    axes[i, 1].set_title('Predicted Mask')
    axes[i, 1].axis('off')
    
    # Overlay
    overlay = cv2.addWeighted(full_images[idx, :, :, 0], 0.7,
                              binary_prediction[:, :, 0], 0.3, 0)
    axes[i, 2].imshow(overlay, cmap='gray')
    axes[i, 2].set_title('Overlay')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show() 
