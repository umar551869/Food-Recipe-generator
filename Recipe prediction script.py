import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import textwrap

def load_model_and_data():
    """
    Load the trained model, label encoder, and recipe data
    """
    print("Loading model and data...")
    
    # Load the model
    if not os.path.exists('food_recipe_matcher.h5'):
        print("Error: Model file 'food_recipe_matcher.h5' not found!")
        return None, None, None
    
    model = tf.keras.models.load_model('food_recipe_matcher.h5')
    
    # Load the label encoder
    if not os.path.exists('label_encoder.pkl'):
        print("Error: Label encoder file 'label_encoder.pkl' not found!")
        return model, None, None
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load the recipe data
    if not os.path.exists('recipe_data.pkl'):
        print("Error: Recipe data file 'recipe_data.pkl' not found!")
        return model, label_encoder, None
    
    recipe_data = pd.read_pickle('recipe_data.pkl')
    
    print("Model and data loaded successfully.")
    return model, label_encoder, recipe_data

def predict_recipe(image_path, model, label_encoder, recipe_data, top_n=3, img_size=(224, 224)):
    """
    Predict the recipe from the given image
    """
    try:
        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found!")
            return None
        
        # Load and preprocess the image
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Get top N predictions
        top_indices = np.argsort(predictions[0])[-top_n:][::-1]
        top_probabilities = [predictions[0][i] for i in top_indices]
        
        # Get recipe titles
        top_recipe_indices = label_encoder.inverse_transform(top_indices)
        
        # Get recipe details
        results = []
        for idx, prob in zip(top_recipe_indices, top_probabilities):
            recipe = recipe_data[recipe_data['Title'] == idx].iloc[0]
            
            # Extract ingredients
            try:
                if isinstance(recipe['Ingredients'], str):
                    try:
                        ingredients = eval(recipe['Ingredients'])
                    except:
                        ingredients = recipe['Ingredients']
                else:
                    ingredients = recipe['Ingredients']
            except:
                ingredients = "Could not parse ingredients"
            
            results.append({
                'title': recipe['Title'],
                'confidence': float(prob),
                'ingredients': ingredients,
                'instructions': recipe['Instructions']
            })
        
        return results
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def format_ingredients(ingredients):
    """Format ingredients for display"""
    if isinstance(ingredients, list):
        return "\n".join([f"• {ing}" for ing in ingredients])
    else:
        return str(ingredients)

def display_prediction(image_path, prediction_results):
    """
    Display the image and the top predicted recipe
    """
    if not prediction_results:
        print("No prediction results to display.")
        return
    
    # Load the image
    img = Image.open(image_path)
    
    # Get the top prediction
    top_result = prediction_results[0]
    
    plt.figure(figsize=(12, 8))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')
    
    # Display the recipe
    plt.subplot(1, 2, 2)
    recipe_text = f"Recipe: {top_result['title']}\n"
    recipe_text += f"Confidence: {top_result['confidence']:.2f}\n\n"
    recipe_text += "INGREDIENTS:\n"
    recipe_text += format_ingredients(top_result['ingredients'])
    
    # Truncate instructions if too long
    instructions = str(top_result['instructions'])
    if len(instructions) > 1000:
        instructions = instructions[:1000] + "..."
    
    recipe_text += "\n\nINSTRUCTIONS:\n"
    recipe_text += instructions
    
    plt.text(0, 0.5, textwrap.fill(recipe_text, width=60), 
             verticalalignment='center', fontsize=9, wrap=True)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.close()
    
    print(f"Recipe: {top_result['title']}")
    print(f"Confidence: {top_result['confidence']:.2f}")
    print("\nINGREDIENTS:")
    print(format_ingredients(top_result['ingredients']))
    print("\nINSTRUCTIONS:")
    print(instructions)
    print("\nVisualization saved as 'prediction_result.png'")

def main():
    parser = argparse.ArgumentParser(description='Predict recipe from food image')
    parser.add_argument('image_path', type=str, help='Path to the food image')
    args = parser.parse_args()
    
    # Load model and data
    model, label_encoder, recipe_data = load_model_and_data()
    if model is None or label_encoder is None or recipe_data is None:
        return
    
    # Make prediction
    prediction_results = predict_recipe(args.image_path, model, label_encoder, recipe_data)
    if prediction_results is None:
        return
    
    # Display prediction
    display_prediction(args.image_path, prediction_results)

if __name__ == "__main__":
    main()