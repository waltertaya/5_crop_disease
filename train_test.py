import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, TimeDistributed, LSTM, Input, Reshape, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import cv2
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration parameters
IMG_SIZE = 224  # Resize images to 224x224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
MAX_IMAGES_PER_CLASS = 150  # Limit to 150 images per disease class

# Set your base directory path here
BASE_DIR = "/kaggle/input/five-crop-diseases-dataset/Crop Diseases Dataset/Crop Diseases/Crop___Disease"  # CHANGE THIS TO YOUR DIRECTORY PATH

# Function to create dataset
def create_dataset(base_dir):
    """
    Creates datasets from a directory structure where:
    base_dir/crop_name/disease_name/images
    
    Returns:
    - X: images
    - y: labels (one-hot encoded)
    - class_names: list of class names
    """
    X = []  # Store images
    y = []  # Store labels
    class_names = []  # Store class names
    
    # Get all crop folders
    crops = os.listdir(base_dir)
    
    label_idx = 0
    label_map = {}
    
    print("Loading and preprocessing images...")
    
    # Loop through crop types
    for crop in crops:
        crop_path = os.path.join(base_dir, crop)
        
        # Skip if not a directory
        if not os.path.isdir(crop_path):
            continue
            
        # Get disease folders for this crop
        diseases = os.listdir(crop_path)
        
        # Loop through diseases
        for disease in diseases:
            disease_path = os.path.join(crop_path, disease)
            
            # Skip if not a directory
            if not os.path.isdir(disease_path):
                continue
            
            # Create class name as "crop_disease"
            class_name = f"{crop}_{disease}"
            class_names.append(class_name)
            label_map[class_name] = label_idx
            label_idx += 1
            
            # Get all images in this disease folder
            image_files = os.listdir(disease_path)
            
            # Limit to MAX_IMAGES_PER_CLASS images per class
            if len(image_files) > MAX_IMAGES_PER_CLASS:
                # Randomly select MAX_IMAGES_PER_CLASS images
                image_files = np.random.choice(image_files, MAX_IMAGES_PER_CLASS, replace=False)
            
            print(f"Processing {len(image_files)} images for {class_name} (limited to {MAX_IMAGES_PER_CLASS})")
            
            # Loop through and process images
            for img_file in tqdm(image_files):
                img_path = os.path.join(disease_path, img_file)
                
                # Only process image files
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert BGR to RGB (OpenCV loads as BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize image
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                # Normalize pixel values
                img = img / 255.0
                
                X.append(img)
                y.append(label_map[class_name])
    
    # Convert lists to arrays
    X = np.array(X)
    y = np.array(y)
    
    # One-hot encode labels
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=len(class_names))
    
    print(f"Dataset created with {len(X)} images and {len(class_names)} classes")
    
    return X, y_onehot, class_names, label_map

# Create dataset
X, y, class_names, label_map = create_dataset(BASE_DIR)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create a function to visualize some augmented samples
def visualize_augmentation(X_sample, y_sample, datagen, class_names):
    """Visualize augmented samples"""
    # Get a sample image
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(2, 5, 1)
    plt.imshow(X_sample)
    class_idx = np.argmax(y_sample)
    plt.title(f"Original: {class_names[class_idx]}")
    plt.axis('off')
    
    # Generate augmented images
    i = 2
    for batch in datagen.flow(X_sample.reshape(1, IMG_SIZE, IMG_SIZE, 3), batch_size=1):
        plt.subplot(2, 5, i)
        plt.imshow(batch[0])
        plt.title(f"Augmented #{i-1}")
        plt.axis('off')
        i += 1
        if i > 10:
            break
    
    plt.tight_layout()
    plt.show()

# Visualize augmentation on a sample
sample_idx = np.random.randint(0, len(X_train))
visualize_augmentation(X_train[sample_idx], y_train[sample_idx], train_datagen, class_names)

# Build a CNN-LSTM model
def build_cnn_lstm_model(input_shape, num_classes):
    """Build and return a CNN-LSTM model"""
    
    # Use MobileNetV2 as base model with pre-trained weights
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create the model
    model = Sequential()
    
    # Add the base model
    model.add(base_model)
    
    # Add GlobalAveragePooling to reduce dimensions
    model.add(GlobalAveragePooling2D())
    
    # Reshape for LSTM (treating features as time steps)
    model.add(Reshape((1, -1)))  # Reshape to (1, features)
    
    # Add LSTM layers
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    
    # Add Dense layers for classification
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Build model
model = build_cnn_lstm_model((IMG_SIZE, IMG_SIZE, 3), len(class_names))
model.summary()

# Set up callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

model_checkpoint = ModelCheckpoint(
    'best_crop_disease_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

callbacks = [early_stopping, reduce_lr, model_checkpoint]

# Train the model
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training history
def plot_history(history):
    """Plot training and validation accuracy and loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Loss plot
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Generate confusion matrix
plt.figure(figsize=(15, 12))
conf_mat = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Create a function to predict disease for a single image
def predict_disease(image_path, model, class_names):
    """
    Predict the disease from an image
    
    Args:
        image_path: Path to the image file
        model: Trained model
        class_names: List of class names
        
    Returns:
        Predicted class name and confidence
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return "Error", "Could not read image", 0.0
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Predict
    predictions = model.predict(img)
    
    # Get top prediction
    top_pred_idx = np.argmax(predictions[0])
    top_pred_conf = predictions[0][top_pred_idx]
    
    # Get class name
    predicted_class = class_names[top_pred_idx]
    
    # Parse crop and disease from class name
    parts = predicted_class.split('_', 1)
    crop = parts[0]
    disease = parts[1] if len(parts) > 1 else "Unknown"
    
    return crop, disease, top_pred_conf

# Function to visualize the prediction
def visualize_prediction(image_path, model, class_names):
    """Visualize the prediction for a single image"""
    # Get prediction
    crop, disease, confidence = predict_disease(image_path, model, class_names)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display image with prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Crop: {crop}\nDisease: {disease}\nConfidence: {confidence:.2f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Select and display predictions for some test images
def display_test_samples(X_test, y_test, model, class_names, num_samples=5):
    """Display predictions for random test samples"""
    # Get random indices
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, num_samples * 3))
    
    for i, idx in enumerate(indices):
        # Get image and true label
        img = X_test[idx]
        true_class_idx = np.argmax(y_test[idx])
        true_class = class_names[true_class_idx]
        
        # Get prediction
        pred = model.predict(np.expand_dims(img, axis=0))
        pred_class_idx = np.argmax(pred[0])
        pred_class = class_names[pred_class_idx]
        confidence = pred[0][pred_class_idx]
        
        # Parse crop and disease
        true_crop, true_disease = true_class.split('_', 1)
        pred_crop, pred_disease = pred_class.split('_', 1)
        
        # Plot
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img)
        title = f"True: {true_crop} - {true_disease}\n"
        title += f"Pred: {pred_crop} - {pred_disease} ({confidence:.2f})"
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Display predictions for some test samples
display_test_samples(X_test, y_test, model, class_names, num_samples=5)

# Create a class for the crop disease prediction system
class CropDiseasePredictor:
    def __init__(self, model=None, model_path=None, class_names=None):
        """
        Initialize the predictor
        
        Args:
            model: Trained model (if already loaded)
            model_path: Path to saved model (if not already loaded)
            class_names: List of class names
        """
        self.img_size = IMG_SIZE
        
        # Set class names
        self.class_names = class_names
        
        # Load model
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Either model or model_path must be provided")
    
    def predict(self, image_path):
        """
        Predict disease from image path
        
        Args:
            image_path: Path to image file
            
        Returns:
            crop, disease, confidence
        """
        return predict_disease(image_path, self.model, self.class_names)
    
    def predict_from_image(self, img):
        """
        Predict disease from a loaded image
        
        Args:
            img: Image as numpy array (BGR format from cv2)
            
        Returns:
            crop, disease, confidence
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = self.model.predict(img)
        
        # Get top prediction
        top_pred_idx = np.argmax(predictions[0])
        top_pred_conf = predictions[0][top_pred_idx]
        
        # Get class name
        predicted_class = self.class_names[top_pred_idx]
        
        # Parse crop and disease from class name
        parts = predicted_class.split('_', 1)
        crop = parts[0]
        disease = parts[1] if len(parts) > 1 else "Unknown"
        
        return crop, disease, top_pred_conf
    
    def visualize_prediction(self, image_path):
        """
        Visualize prediction for an image
        
        Args:
            image_path: Path to image file
        """
        visualize_prediction(image_path, self.model, self.class_names)
        
    def predict_from_folder(self, folder_path):
        """
        Predict diseases for all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            DataFrame with predictions
        """
        results = []
        
        # Get all image files
        files = [f for f in os.listdir(folder_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
        # Make predictions
        for file in tqdm(files):
            file_path = os.path.join(folder_path, file)
            crop, disease, confidence = self.predict(file_path)
            
            results.append({
                'Image': file,
                'Crop': crop,
                'Disease': disease,
                'Confidence': confidence
            })
        
        # Create DataFrame
        return pd.DataFrame(results)

# Create predictor instance using the trained model
predictor = CropDiseasePredictor(model=model, class_names=class_names)

# Save the model for later use
model.save('crop_disease_cnn_lstm_model.keras')
print(f"Model saved as 'crop_disease_cnn_lstm_model.keras'")

# Create a sample UI using widgets (for Jupyter notebooks)
try:
    from ipywidgets import widgets
    from IPython.display import display, clear_output
    
    def upload_and_predict():
        """Create a simple file upload widget for prediction"""
        upload_widget = widgets.FileUpload(
            accept='.jpg,.jpeg,.png',
            multiple=False,
            description="Upload Image"
        )
        
        output = widgets.Output()
        
        def on_upload_change(change):
            if not change.new:
                return
                
            with output:
                clear_output()
                for name, file_info in change.new.items():
                    # Save temporary file
                    with open(name, 'wb') as f:
                        f.write(file_info['content'])
                        
                    # Predict
                    crop, disease, confidence = predictor.predict(name)
                    
                    # Display image and prediction
                    img = cv2.imread(name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(8, 8))
                    plt.imshow(img)
                    plt.title(f"Crop: {crop}\nDisease: {disease}\nConfidence: {confidence:.2f}")
                    plt.axis('off')
                    plt.show()
                    
                    # Remove temporary file
                    os.remove(name)
        
        upload_widget.observe(on_upload_change, names='value')
        
        # Display widgets
        display(upload_widget)
        display(output)
        
    print("Interactive prediction widget available, run upload_and_predict() to use.")
except ImportError:
    print("ipywidgets not available, skipping interactive widget.")

# Create a function for testing with randomly selected test images
def test_with_random_samples(num_samples=5):
    """Test the model with randomly selected test images"""
    # Get random indices
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        print(f"Sample {i+1}:")
        
        # Get image and true label
        img = X_test[idx]
        true_class_idx = np.argmax(y_test[idx])
        true_class = class_names[true_class_idx]
        
        # Parse true crop and disease
        true_crop, true_disease = true_class.split('_', 1)
        
        # Add the batch dimension for prediction
        img_batch = np.expand_dims(img, axis=0)
        
        # Make prediction
        pred = model.predict(img_batch)
        pred_class_idx = np.argmax(pred[0])
        pred_class = class_names[pred_class_idx]
        confidence = pred[0][pred_class_idx]
        
        # Parse predicted crop and disease
        pred_crop, pred_disease = pred_class.split('_', 1)
        
        # Display results
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        title = f"True: {true_crop} - {true_disease}\n"
        title += f"Predicted: {pred_crop} - {pred_disease}\n"
        title += f"Confidence: {confidence:.2f}"
        plt.title(title)
        plt.axis('off')
        plt.show()
        
        print(f"True: {true_crop} - {true_disease}")
        print(f"Predicted: {pred_crop} - {pred_disease}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 50)

# Function to make predictions on a specific image
def predict_on_image():
    """
    Allow user to input a path and make prediction
    This is a placeholder and should be adapted for actual usage
    """
    print("To predict on a specific image, use:")
    print("crop, disease, confidence = predictor.predict('/path/to/your/image.jpg')")
    print("# Example usage:")
    print("image_path = '/path/to/your/image.jpg'  # Replace with actual path")
    print("crop, disease, confidence = predictor.predict(image_path)")
    print("print(f'Crop: {crop}')")
    print("print(f'Disease: {disease}')")
    print("print(f'Confidence: {confidence:.2f}')")
    print("\nTo visualize the prediction:")
    print("predictor.visualize_prediction(image_path)")

# Instructions for using the model
print("\n" + "="*50)
print("Instructions for using the Crop Disease Prediction Model:")
print("="*50)
print("1. The model has been trained and evaluated.")
print("2. Use test_with_random_samples() to view predictions on random test images.")
print("3. To make predictions on a specific image:")
print("   crop, disease, confidence = predictor.predict('/path/to/your/image.jpg')")
print("4. To visualize a prediction:")
print("   predictor.visualize_prediction('/path/to/your/image.jpg')")
print("5. The model has been saved as 'crop_disease_cnn_lstm_model.keras'")
print("6. To load the model in the future:")
print("   loaded_model = tf.keras.models.load_model('crop_disease_cnn_lstm_model.keras')")
print("   predictor = CropDiseasePredictor(model=loaded_model, class_names=class_names)")
print("="*50)

# Run test with random samples
test_with_random_samples(3)