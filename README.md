# Food-Recipe-generator
Food Image to Recipe Generator
This repository contains a Jupyter Notebook (food-recipe.ipynb) that implements a deep learning model to generate recipes from food images using a combination of Vision Transformer (ViT) and GPT-2 models. The project is designed to run on Kaggle with GPU support, leveraging PyTorch and the Hugging Face Transformers library.
Overview
The notebook trains a model to generate detailed recipes, including titles, ingredients, and instructions, based on food images. It includes functionality for data preparation, model training, inference, and memory management on GPU environments. The model combines ViT for image feature extraction and GPT-2 for text generation, fine-tuned on a dataset of food images and corresponding recipes.
Features

Data Preparation: Processes a CSV file with recipe metadata and matches images from a specified directory, handling various image extensions.
Model Architecture: Combines ViT (google/vit-base-patch16-224) for image processing and GPT-2 for recipe generation, with a linear projection layer to align image and text embeddings.
Training: Supports multi-epoch training with gradient accumulation, mixed precision, and learning rate scheduling. Saves checkpoints and the best model based on validation loss.
Inference: Generates recipes for selected images with post-processing to ensure structured output (title, ingredients, instructions).
Memory Management: Includes functions to monitor and free GPU memory, crucial for running on Kaggle's GPU resources.
Output Handling: Saves generated recipes and training metrics (loss plots) to an output directory and optionally creates a Kaggle dataset.

Requirements
To run the notebook, you need the following dependencies, which are typically pre-installed in Kaggle environments:

Python 3.11
PyTorch
Transformers (Hugging Face)
Pandas
NumPy
PIL (Pillow)
Matplotlib
TQDM
Kaggle API (optional, for dataset creation)

Install additional dependencies if running outside Kaggle:
pip install torch torchvision transformers pandas numpy pillow matplotlib tqdm

Dataset
The notebook expects a dataset with the following structure:

A CSV file (recipe.csv) containing columns: Image_Name, Title, Ingredients, Instructions.
A directory (Food Images/Food Images) containing images referenced in the CSV.

The dataset used in the notebook is assumed to be available at /kaggle/input/recipe/. You can replace this with your own dataset, ensuring the same structure.
Usage

Setup:

Clone this repository or download food-recipe.ipynb.
Ensure the dataset is available in the expected paths or update the paths in the notebook.
If running locally, install dependencies and ensure GPU support (if available).


Running the Notebook:

Open food-recipe.ipynb in Jupyter or Kaggle.
Execute the cells sequentially. The notebook includes:
Data preparation and dataset creation.
Model training with checkpointing.
Inference on selected images.
Optional Kaggle dataset creation for outputs.




Training:

The train_model function trains the model for a specified number of epochs (default: 10) using 60% of the dataset.
Checkpoints are saved in /kaggle/working/food_recipe_model/, with only the latest checkpoint retained to manage disk space.


Inference:

The run_inference function generates recipes for a specified number of images (default: 5).
Results are saved to /kaggle/working/food_recipe_model/inference_results.txt.


Output:

Training produces a loss plot (loss_plot.png) and model checkpoints (final_model.pt, best_model.pt).
Inference results are saved as text files with generated recipes.



Example
To run inference on 5 images:
run_inference(
    csv_path='/kaggle/input/recipe/recipe.csv',
    image_dir='/kaggle/input/recipe/Food Images/Food Images',
    model_path='/kaggle/working/food_recipe_model/final_model.pt',
    output_dir='/kaggle/working/food_recipe_model',
    num_images=5
)

This will select 5 diverse images, generate recipes, and save the results to inference_results.txt.
Notes

GPU Usage: The notebook includes memory management functions (free_gpu_memory, print_gpu_memory) to prevent out-of-memory errors on Kaggle GPUs.
Post-Processing: The post_process_recipe function ensures generated recipes have a consistent format, with fallback to simulated recipes for specific cases (e.g., images with "turkey", "pork", or "shrimp").
Kaggle Integration: The notebook supports creating a Kaggle dataset from the output directory using the Kaggle API. Ensure API credentials are configured if running on Kaggle.
Error Handling: The code includes robust error handling for image loading, model training, and file operations.

Limitations

The model may generate incomplete or nonsensical recipes if the training dataset is limited or the image quality is poor.
GPU memory constraints on Kaggle may limit batch size or dataset size. Adjust batch_size or dataset_percentage if needed.
The simulated recipe fallback is triggered for specific keywords to handle cases where the model output is suboptimal.

Contributing
Contributions are welcome! Please submit issues or pull requests for bug fixes, optimizations, or new features.
License
This project is licensed under the MIT License. See the LICENSE file for details.
