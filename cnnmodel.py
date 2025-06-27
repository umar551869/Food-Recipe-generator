import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set memory growth for GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth setting error: {e}")

# Function to load and preprocess the data
def preprocess_data(csv_path, img_folder):
    print("Loading and preprocessing data...")
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Check if image folder exists
    if not os.path.exists(img_folder):
        raise FileNotFoundError(f"Image folder not found: {img_folder}")
    
    # Load recipe data
    df = pd.read_csv(csv_path)
    
    # Clean up column names and data
    df.columns = df.columns.str.strip()
    
    # Check required columns
    required_columns = ['Title', 'Ingredients', 'Instructions']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")
    
    # Check if 'Image_Name' is in the dataframe
    if 'Image_Name' not in df.columns:
        # Try alternative column name
        image_col = [col for col in df.columns if 'image' in col.lower()]
        if image_col:
            df.rename(columns={image_col[0]: 'Image_Name'}, inplace=True)
        else:
            raise ValueError("Could not find image filename column in CSV")
    
    # Handle missing values
    for col in required_columns + ['Image_Name']:
        if df[col].isna().any():
            print(f"Warning: Found {df[col].isna().sum()} missing values in '{col}' column")
            # For Image_Name, we'll drop rows with missing values
            if col == 'Image_Name':
                df = df.dropna(subset=[col])
            # For other columns, fill with placeholder text
            else:
                df[col] = df[col].fillna(f"No {col.lower()} available")
    
    print(f"CSV loaded with {len(df)} recipes")
    
    # Verify recipes with available images
    available_images = set(os.listdir(img_folder))
    df['image_exists'] = False  # Initialize column
    
    # Check multiple image formats
    for idx, row in df.iterrows():
        img_name = row['Image_Name']
        # Check if the image exists with various extensions or without extension
        for ext in ['', '.jpg', '.jpeg', '.png']:
            if (img_name + ext) in available_images or os.path.exists(os.path.join(img_folder, img_name + ext)):
                df.at[idx, 'image_exists'] = True
                break
    
    # Filter recipes with available images
    df_with_images = df[df['image_exists']].copy()
    
    if len(df_with_images) == 0:
        raise ValueError("No matching images found in the specified folder. Check image filenames and folder path.")
    
    print(f"Found {len(df_with_images)} recipes with matching images")
    
    # Encode recipe titles as numerical labels
    label_encoder = LabelEncoder()
    df_with_images['label'] = label_encoder.fit_transform(df_with_images['Title'])
    
    return df_with_images, label_encoder

# Function to load and prepare images
def load_images(df, img_folder, img_size=(224, 224), batch_size=32):
    print("Loading images...")
    
    img_paths = []
    labels = []
    
    for idx, row in df.iterrows():
        img_name = row['Image_Name']
        
        # Check for image with different possible extensions
        found = False
        for ext in ['', '.jpg', '.jpeg', '.png']:
            img_path = os.path.join(img_folder, img_name + ext)
            if os.path.exists(img_path):
                img_paths.append(img_path)
                labels.append(row['label'])
                found = True
                break
        
        if not found:
            print(f"Warning: Could not find image for recipe '{row['Title']}' with Image_Name '{img_name}'")
    
    if not img_paths:
        raise ValueError("No valid image paths found")
        
    num_classes = df['label'].nunique()
    
    # Create a data generator to load images in batches
    def data_generator(img_paths, labels, batch_size):
        num_samples = len(img_paths)
        if num_samples == 0:
            raise ValueError("No images to generate batches from")
            
        while True:
            indices = np.random.permutation(num_samples)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]
                
                batch_x = np.zeros((len(batch_indices), img_size[0], img_size[1], 3))
                batch_y = np.zeros((len(batch_indices), num_classes))
                
                for i, idx in enumerate(batch_indices):
                    try:
                        img = load_img(img_paths[idx], target_size=img_size)
                        img_array = img_to_array(img)
                        img_array = preprocess_input(img_array)
                        batch_x[i] = img_array
                        batch_y[i] = to_categorical(labels[idx], num_classes=num_classes)
                    except Exception as e:
                        print(f"Error loading image {img_paths[idx]}: {e}")
                        # Use a blank image
                        batch_x[i] = np.zeros((img_size[0], img_size[1], 3))
                        batch_y[i] = to_categorical(labels[idx], num_classes=num_classes)
                
                yield batch_x, batch_y
    
    return img_paths, labels, num_classes, data_generator

# Function to build the 3-layer CNN model
def build_model(input_shape, num_classes):
    print(f"Building 3-layer CNN model for {num_classes} recipe classes...")
    
    if num_classes <= 1:
        raise ValueError("Need at least 2 classes to build a classification model")
    
    inputs = Input(shape=input_shape)
    
    # Layer 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    # Layer 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Layer 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Classifier
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to train the model
def train_model(model, train_gen, val_gen, train_steps, val_steps, epochs=20):
    print("Training model...")
    
    # Validate parameters
    if train_steps <= 0:
        train_steps = 1
        print("Warning: train_steps was <= 0, setting to 1")
    
    if val_steps <= 0:
        val_steps = 1
        print("Warning: val_steps was <= 0, setting to 1")
    
    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Add model checkpoint to save best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train the model
    try:
        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        return history
    except Exception as e:
        print(f"Error during training: {e}")
        raise

# Function to plot training history
def plot_history(history):
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("Training history plot saved as 'training_history.png'")
    except Exception as e:
        print(f"Error creating training history plot: {e}")

# Function to predict recipe from image
def predict_recipe(model, image_path, label_encoder, df, img_size=(224, 224)):
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get recipe details
        recipe_title = label_encoder.inverse_transform([predicted_class])[0]
        recipe_details = df[df['Title'] == recipe_title]
        
        if len(recipe_details) == 0:
            return {
                'title': recipe_title,
                'confidence': float(confidence),
                'ingredients': "Recipe details not found",
                'instructions': "Recipe details not found"
            }
        
        recipe_details = recipe_details.iloc[0]
        
        return {
            'title': recipe_title,
            'confidence': float(confidence),
            'ingredients': recipe_details['Ingredients'],
            'instructions': recipe_details['Instructions']
        }
    except Exception as e:
        print(f"Error predicting recipe: {e}")
        raise

# Main function
def main():
    try:
        # Paths and parameters
        csv_path = 'recipe.csv'  # Update with your actual CSV path
        img_folder = 'Food Images/Food Images'  # Update with your actual image folder path
        batch_size = 16  # Small batch size for memory efficiency
        img_size = (224, 224)
        epochs = 20
        
        # Check if paths exist
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}")
            return
        
        if not os.path.exists(img_folder):
            print(f"Error: Image folder not found at {img_folder}")
            return
        
        # Preprocess data
        df, label_encoder = preprocess_data(csv_path, img_folder)
        
        # Check if we have enough unique recipes
        if df['label'].nunique() < 2:
            print("Error: Need at least 2 unique recipes with images for classification")
            return
        
        # Split data into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        # Load and prepare images
        train_paths, train_labels, num_classes, data_generator = load_images(train_df, img_folder, img_size, batch_size)
        val_paths, val_labels, _, _ = load_images(val_df, img_folder, img_size, batch_size)
        
        # Calculate steps per epoch
        train_steps = max(1, len(train_paths) // batch_size)
        val_steps = max(1, len(val_paths) // batch_size)
        
        # Create data generators
        train_gen = data_generator(train_paths, train_labels, batch_size)
        val_gen = data_generator(val_paths, val_labels, batch_size)
        
        # Build model
        model = build_model((img_size[0], img_size[1], 3), num_classes)
        model.summary()
        
        # Train model
        history = train_model(model, train_gen, val_gen, train_steps, val_steps, epochs)
        
        # Plot training history
        plot_history(history)
        
        # Save model
        model.save('food_recipe_matcher.h5')
        print("Model saved as 'food_recipe_matcher.h5'")
        
        # Save label encoder and recipe data for inference
        import pickle
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        print("Label encoder saved as 'label_encoder.pkl'")
        
        # Save a smaller version of the dataframe with just necessary columns
        df_small = df[['Title', 'Ingredients', 'Instructions', 'Image_Name', 'label']].copy()
        df_small.to_pickle('recipe_data.pkl')
        print("Recipe data saved as 'recipe_data.pkl'")
        
        print("Model training complete.")
        
        # Example prediction
        if len(train_paths) > 0:
            print("\nExample prediction:")
            sample_img_path = train_paths[0]
            result = predict_recipe(model, sample_img_path, label_encoder, df)
            print(f"Predicted recipe: {result['title']} (Confidence: {result['confidence']:.2f})")
    
    except Exception as e:
        print(f"An error occurred in the main function: {e}")

# Example of using the trained model with a new image
def use_model(image_path):
    try:
        # Check if model exists
        if not os.path.exists('food_recipe_matcher.h5'):
            print("Error: Model file 'food_recipe_matcher.h5' not found. Please train the model first.")
            return None
            
        # Check if necessary files exist
        for file in ['label_encoder.pkl', 'recipe_data.pkl']:
            if not os.path.exists(file):
                print(f"Error: Required file '{file}' not found. Please train the model first.")
                return None
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None
        
        # Load model and necessary data
        model = tf.keras.models.load_model('food_recipe_matcher.h5')
        
        import pickle
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        df = pd.read_pickle('recipe_data.pkl')
        
        # Predict recipe
        result = predict_recipe(model, image_path, label_encoder, df)
        
        print(f"Recipe: {result['title']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("\nIngredients:")
        print(result['ingredients'])
        print("\nInstructions:")
        print(result['instructions'])
        
        return result
    
    except Exception as e:
        print(f"An error occurred when using the model: {e}")
        return None

if __name__ == "__main__":
    main()