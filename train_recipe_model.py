import os
import pandas as pd
from PIL import Image, ImageFile
from rarfile import RarFile
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
import logging
import psutil
import re

# Setup
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.DEBUG, filename='train.log', filemode='w')
logger = logging.getLogger(__name__)

# Preprocessing
def preprocess_data(recipe_csv_path, rar_path, image_dir, output_pkl='processed_recipes.pkl', sample_fraction=0.2):
    logger.info("Starting preprocessing...")
    if not os.path.exists(recipe_csv_path) or not os.path.exists(rar_path):
        raise FileNotFoundError("Recipe CSV or RAR file missing")

    def extract_images(rar_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with RarFile(rar_path) as rf:
            for f in tqdm(rf.infolist(), desc="Extracting images"):
                rf.extract(f, output_dir)
        logger.info(f"Images extracted to {output_dir}")

    if not os.path.exists(image_dir):
        extract_images(rar_path, image_dir)

    df = pd.read_csv(recipe_csv_path, encoding='latin1', on_bad_lines='skip', engine='python')
    df.dropna(subset=['Title', 'Ingredients', 'Instructions', 'Image_Name'], inplace=True)
    df['Cleaned_Ingredients'] = df['Ingredients'].str.replace('[^\\w\\s,.-]', '', regex=True)
    df['Instructions'] = df['Instructions'].str.replace('\\s+', ' ', regex=True)
    df = df[df['Cleaned_Ingredients'].str.len() > 10]
    df = df[df['Instructions'].str.len() > 20]
    logger.info(f"Filtered dataset: {len(df)}")

    def normalize_image_name(name):
        if not isinstance(name, str):
            return None
        name = re.sub(r'[^\\w\\s-]', '', name.strip().lower()).replace(' ', '_')
        return name.split('.')[0]

    df['Image_Name_Norm'] = df['Image_Name'].apply(normalize_image_name)

    image_index = {}
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                norm_name = normalize_image_name(file)
                image_index[norm_name] = os.path.join(root, file)

    def resolve_image_path(image_name):
        norm_name = normalize_image_name(image_name)
        return image_index.get(norm_name)

    df['image_path'] = df['Image_Name'].apply(resolve_image_path)
    df = df.dropna(subset=['image_path']).reset_index(drop=True)
    logger.info(f"Dataset with images: {len(df)}")

    df['full_text'] = "TITLE: " + df['Title'] + " INGREDIENTS: " + df['Cleaned_Ingredients'] + " INSTRUCTIONS: " + df['Instructions']

    os.makedirs("preprocessed_images", exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    new_paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing images"):
        try:
            with Image.open(row['image_path']) as img:
                img = img.convert('RGB')
                img = transform(img)
                new_path = f"preprocessed_images/{idx}.pt"
                torch.save(img, new_path)
                new_paths.append(new_path)
        except Exception as e:
            logger.error(f"Error processing {row['image_path']}: {e}")
            new_paths.append(None)
    df['preprocessed_image_path'] = new_paths
    df = df.dropna(subset=['preprocessed_image_path']).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    encodings = []
    for text in tqdm(df['full_text'], desc="Tokenizing text"):
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        encodings.append({
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        })
    df['encoding'] = encodings

    df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
    logger.info(f"Sampled dataset: {len(df)}")
    df.to_pickle(output_pkl)
    return df

# Model Definition
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super(ImageEncoder, self).__init__()
        self.mobilenet = mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, embed_dim)

    def forward(self, x):
        return self.mobilenet(x)

class RecipeGenerator(nn.Module):
    def __init__(self, embed_dim=128):
        super(RecipeGenerator, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.transformer = AutoModelForCausalLM.from_pretrained('distilgpt2')
        self.transformer.transformer.h = self.transformer.transformer.h[:4]
        self.image_projection = nn.Linear(embed_dim, self.transformer.config.n_embd)
        self.vocab_size = self.transformer.config.vocab_size
        logger.info(f"Initialized RecipeGenerator with vocab_size: {self.vocab_size}")

        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, image_embed, input_ids, attention_mask):
        if image_embed.dim() != 2 or image_embed.size(0) != input_ids.size(0):
            raise ValueError(f"Expected image_embed shape (batch_size, embed_dim), got {image_embed.shape}")
        if input_ids.dim() != 2 or attention_mask.dim() != 2:
            raise ValueError(f"Expected input_ids and attention_mask shape (batch_size, seq_len), got {input_ids.shape}, {attention_mask.shape}")

        image_embed = self.image_projection(image_embed)
        input_embeds = self.transformer.transformer.wte(input_ids)
        combined_embeds = torch.cat([image_embed.unsqueeze(1), input_embeds], dim=1)
        extended_attention_mask = torch.cat([
            torch.ones((attention_mask.size(0), 1), device=attention_mask.device),
            attention_mask
        ], dim=1)
        
        outputs = self.transformer(inputs_embeds=combined_embeds, attention_mask=extended_attention_mask)
        logits = outputs.logits
        logger.debug(f"Forward logits shape: {logits.shape}, expected: ({input_ids.size(0)}, {input_ids.size(1) + 1}, {self.vocab_size})")
        return logits

class ImageToRecipeModel(nn.Module):
    def __init__(self, embed_dim=128):
        super(ImageToRecipeModel, self).__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.recipe_generator = RecipeGenerator(embed_dim)

        for param in self.parameters():
            param.requires_grad = True
        for param in self.image_encoder.mobilenet.parameters():
            param.requires_grad = False
        for param in self.recipe_generator.transformer.parameters():
            param.requires_grad = False
        for param in self.image_encoder.mobilenet.classifier.parameters():
            param.requires_grad = True
        for param in self.recipe_generator.image_projection.parameters():
            param.requires_grad = True

    def forward(self, image, input_ids, attention_mask):
        image_embed = self.image_encoder(image)
        logits = self.recipe_generator(image_embed, input_ids, attention_mask)
        return logits

# Dataset
class RecipeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=64):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = torch.load(row['preprocessed_image_path'])
            if image.shape != (3, 224, 224):
                logger.warning(f"Invalid image shape: {image.shape}")
                image = torch.zeros((3, 224, 224))
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            image = torch.zeros((3, 224, 224))
        encoding = row['encoding']
        return {
            'image': image,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }

# Training
def train_model(model, train_loader, val_loader, optimizer, device, epochs=5, grad_accum_steps=16, checkpoint_path='image_to_recipe_final.pt'):
    best_loss = float('inf')
    model.vocab_size = model.recipe_generator.vocab_size
    logger.info(f"Model vocab_size: {model.vocab_size}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            try:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                if images.shape[0] != input_ids.shape[0]:
                    logger.error(f"Batch size mismatch: images {images.shape}, input_ids {input_ids.shape}")
                    continue

                labels = torch.cat([
                    torch.full((input_ids.size(0), 1), -100, device=device),
                    input_ids
                ], dim=1)
                labels[labels == model.recipe_generator.tokenizer.pad_token_id] = -100

                logits = model(images, input_ids, attention_mask)
                if logits.shape != (input_ids.size(0), input_ids.size(1) + 1, model.vocab_size):
                    logger.error(f"Invalid logits shape: {logits.shape}, expected {(input_ids.size(0), input_ids.size(1) + 1, model.vocab_size)}")
                    continue

                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / grad_accum_steps
                loss.backward()

                if (i + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * grad_accum_steps
                logger.info(f"Batch {i+1}, Loss: {loss.item() * grad_accum_steps:.4f}")

            except RuntimeError as e:
                logger.error(f"Training error: {e}")
                optimizer.zero_grad()
                continue

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                try:
                    images = batch['image'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = torch.cat([
                        torch.full((input_ids.size(0), 1), -100, device=device),
                        input_ids
                    ], dim=1)
                    labels[labels == model.recipe_generator.tokenizer.pad_token_id] = -100
                    logits = model(images, input_ids, attention_mask)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                    val_loss += loss.item()
                except RuntimeError as e:
                    logger.error(f"Validation error: {e}")
                    continue
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            try:
                torch.save(model.state_dict(), checkpoint_path)
                temp_model = ImageToRecipeModel(embed_dim=128).to(device)
                temp_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                logger.info(f"Checkpoint saved and verified: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Checkpoint error: {e}")

    return model

# Main
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    recipe_csv_path = "recipe.csv"
    rar_path = "Food Images.rar"
    image_dir = "Food Images"
    processed_data_path = "processed_recipes.pkl"

    if not os.path.exists(processed_data_path):
        df = preprocess_data(recipe_csv_path, rar_path, image_dir, processed_data_path, sample_fraction=0.2)
    else:
        df = pd.read_pickle(processed_data_path)

    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = RecipeDataset(df, tokenizer)
    train_idx = list(range(int(0.8 * len(dataset))))
    val_idx = list(range(int(0.8 * len(dataset)), len(dataset)))
    train_loader = DataLoader(dataset, batch_size=1, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=1, sampler=SubsetRandomSampler(val_idx))

    model = ImageToRecipeModel(embed_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model = train_model(model, train_loader, val_loader, optimizer, device)