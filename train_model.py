import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# ==============================================================================
# 1. CUSTOM DATASET
# ==============================================================================
class MultiTaskDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image safely
        img_path = row['image_path']
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Extract targets
        targets = {
            'country': torch.tensor(row['country_encoded'], dtype=torch.long),
            'age': torch.tensor(row['age_encoded'], dtype=torch.long),
            'gender': torch.tensor(row['gender_encoded'], dtype=torch.long),
            'ethnicity': torch.tensor(row['ethnicity_encoded'], dtype=torch.long)
        }
        
        return image, targets

# ==============================================================================
# 2. PYTORCH LIGHTNING MODEL ARCHITECTURE
# ==============================================================================
class MultiTaskMacClassifier(pl.LightningModule):
    def __init__(self, num_countries, num_ages, num_genders, num_ethnicities, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        
        # Core Backbone (MobileNetV3 Large)
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Identity() # Strip final layer
        
        # Multi-task Classification Heads
        self.country_head = nn.Linear(in_features, num_countries)
        self.age_head = nn.Linear(in_features, num_ages)
        self.gender_head = nn.Linear(in_features, num_genders)
        self.ethnicity_head = nn.Linear(in_features, num_ethnicities)
        
        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        shared_features = self.backbone(x)
        return {
            'country': self.country_head(shared_features),
            'age': self.age_head(shared_features),
            'gender': self.gender_head(shared_features),
            'ethnicity': self.ethnicity_head(shared_features)
        }

    def training_step(self, batch, batch_idx):
        x, targets = batch
        outputs = self(x)
        
        loss_country = self.criterion(outputs['country'], targets['country'])
        loss_age = self.criterion(outputs['age'], targets['age'])
        loss_gender = self.criterion(outputs['gender'], targets['gender'])
        loss_ethnicity = self.criterion(outputs['ethnicity'], targets['ethnicity'])
        
        # Total combined loss
        total_loss = loss_country + loss_age + loss_gender + loss_ethnicity
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        outputs = self(x)
        
        loss_country = self.criterion(outputs['country'], targets['country'])
        
        # Calculate accuracy for tracking
        preds = torch.argmax(outputs['country'], dim=1)
        acc = (preds == targets['country']).float().mean()
        
        self.log('val_loss', loss_country, on_epoch=True, prog_bar=True)
        self.log('val_acc_epoch', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

# ==============================================================================
# 3. PIPELINE EXECUTION SCRIPT
# ==============================================================================
def run_balanced_training():
    manifest_path = "multitask_manifest.csv"
    print(f"Reading manifest file: {manifest_path}...")
    df = pd.read_csv(manifest_path)
    
    # --------------------------------------------------------------------------
    # MACRO-REGION MAPPING (Option 2 Implementation)
    # --------------------------------------------------------------------------
    print("Mapping independent countries into 11 Global Macro-Regions...")
    
    macro_regions = {
        # South Asia
        'India': 'South Asia', 'Bangladesh': 'South Asia', 'Pakistan': 'South Asia', 
        'Sri Lanka': 'South Asia', 'Nepal': 'South Asia', 'Bhutan': 'South Asia', 
        'Maldives': 'South Asia', 'Afghanistan': 'South Asia',
        
        # East Asia
        'China': 'East Asia', 'Japan': 'East Asia', 'Korea, Republic of': 'East Asia', 
        "Korea, Democratic People's Republic of": 'East Asia', 'Taiwan, Province of China': 'East Asia', 
        'Mongolia': 'East Asia', 'Hong Kong': 'East Asia', 'Macao': 'East Asia',
        
        # Southeast Asia
        'Indonesia': 'Southeast Asia', 'Philippines': 'Southeast Asia', 'Viet Nam': 'Southeast Asia', 
        'Thailand': 'Southeast Asia', 'Myanmar': 'Southeast Asia', 'Malaysia': 'Southeast Asia', 
        'Cambodia': 'Southeast Asia', 'Lao People\'s Democratic Republic': 'Southeast Asia', 
        'Singapore': 'Southeast Asia', 'Timor-Leste': 'Southeast Asia', 'Brunei Darussalam': 'Southeast Asia',
        
        # Central Asia
        'Uzbekistan': 'Central Asia', 'Kazakhstan': 'Central Asia', 'Tajikistan': 'Central Asia', 
        'Kyrgyzstan': 'Central Asia', 'Turkmenistan': 'Central Asia',
        
        # Middle East & North Africa (MENA)
        'Egypt': 'MENA', 'Iran, Islamic Republic of': 'MENA', 'Turkey': 'MENA', 'Iraq': 'MENA', 
        'Saudi Arabia': 'MENA', 'Yemen': 'MENA', 'Syrian Arab Republic': 'MENA', 'Morocco': 'MENA', 
        'Algeria': 'MENA', 'Jordan': 'MENA', 'United Arab Emirates': 'MENA', 'Israel': 'MENA', 
        'Lebanon': 'MENA', 'Palestine, State of': 'MENA', 'Oman': 'MENA', 'Kuwait': 'MENA', 
        'Qatar': 'MENA', 'Bahrain': 'MENA', 'Tunisia': 'MENA', 'Libya': 'MENA',
        
        # Sub-Saharan Africa
        'Nigeria': 'Sub-Saharan Africa', 'Ethiopia': 'Sub-Saharan Africa', 'Congo, The Democratic Republic of the': 'Sub-Saharan Africa', 
        'South Africa': 'Sub-Saharan Africa', 'Tanzania, United Republic of': 'Sub-Saharan Africa', 
        'Kenya': 'Sub-Saharan Africa', 'Uganda': 'Sub-Saharan Africa', 'Sudan': 'Sub-Saharan Africa', 
        'Ghana': 'Sub-Saharan Africa', 'Cameroon': 'Sub-Saharan Africa', 'Mali': 'Sub-Saharan Africa', 
        'Madagascar': 'Sub-Saharan Africa', 'Senegal': 'Sub-Saharan Africa', 'Zimbabwe': 'Sub-Saharan Africa', 
        'Rwanda': 'Sub-Saharan Africa', 'Guinea': 'Sub-Saharan Africa', 'Burundi': 'Sub-Saharan Africa', 
        'Somalia': 'Sub-Saharan Africa', 'Eritrea': 'Sub-Saharan Africa', 'Sierra Leone': 'Sub-Saharan Africa',
        
        # Western & Northern Europe
        'Germany': 'Western Europe', 'United Kingdom': 'Western Europe', 'France': 'Western Europe', 
        'Italy': 'Western Europe', 'Spain': 'Western Europe', 'Netherlands': 'Western Europe', 
        'Belgium': 'Western Europe', 'Sweden': 'Western Europe', 'Austria': 'Western Europe', 
        'Switzerland': 'Western Europe', 'Denmark': 'Western Europe', 'Finland': 'Western Europe', 
        'Norway': 'Western Europe', 'Ireland': 'Western Europe', 'Portugal': 'Western Europe', 'Iceland': 'Western Europe',
        
        # Eastern & Southern Europe
        'Russian Federation': 'Eastern Europe', 'Ukraine': 'Eastern Europe', 'Poland': 'Eastern Europe', 
        'Romania': 'Eastern Europe', 'Czechia': 'Eastern Europe', 'Hungary': 'Eastern Europe', 
        'Belarus': 'Eastern Europe', 'Bulgaria': 'Eastern Europe', 'Serbia': 'Eastern Europe', 
        'Slovakia': 'Eastern Europe', 'Croatia': 'Eastern Europe', 'Bosnia and Herzegovina': 'Eastern Europe', 
        'Moldova, Republic of': 'Eastern Europe', 'Albania': 'Eastern Europe', 'Lithuania': 'Eastern Europe', 
        'Slovenia': 'Eastern Europe', 'Latvia': 'Eastern Europe', 'Estonia': 'Eastern Europe', 'Greece': 'Eastern Europe',
        
        # North America
        'United States': 'North America', 'Canada': 'North America', 'Bermuda': 'North America', 'Greenland': 'North America',
        
        # Latin America & Caribbean
        'Brazil': 'Latin America', 'Mexico': 'Latin America', 'Colombia': 'Latin America', 
        'Argentina': 'Latin America', 'Peru': 'Latin America', 'Venezuela, Bolivarian Republic of': 'Latin America', 
        'Chile': 'Latin America', 'Ecuador': 'Latin America', 'Guatemala': 'Latin America', 
        'Cuba': 'Latin America', 'Haiti': 'Latin America', 'Bolivia, Plurinational State of': 'Latin America', 
        'Dominican Republic': 'Latin America', 'Honduras': 'Latin America', 'Paraguay': 'Latin America', 
        'El Salvador': 'Latin America', 'Nicaragua': 'Latin America', 'Costa Rica': 'Latin America', 
        'Puerto Rico': 'Latin America', 'Panama': 'Latin America', 'Uruguay': 'Latin America', 'Jamaica': 'Latin America',
        
        # Oceania
        'Australia': 'Oceania', 'New Zealand': 'Oceania', 'Papua New Guinea': 'Oceania', 
        'Fiji': 'Oceania', 'Solomon Islands': 'Oceania', 'Vanuatu': 'Oceania', 'Samoa': 'Oceania', 
        'Kiribati': 'Oceania', 'Tonga': 'Oceania', 'Micronesia, Federated States of': 'Oceania'
    }

    # Map the countries to regions
    df['region'] = df['country'].map(macro_regions)
    
    # Drop unmapped data (tiny islands, dependencies, etc. that don't fit well)
    original_size = len(df)
    df = df.dropna(subset=['region']).reset_index(drop=True)
    print(f"Dropped {original_size - len(df)} images from unmapped territories.")
    
    # OVERWRITE the 'country' column with the new 'region' data
    df['country'] = df['region']
    print(f"Dataset successfully compressed to {df['country'].nunique()} Macro-Regions!\n")
    # --------------------------------------------------------------------------

    # Encode categorical targets
    encoders = {}
    for col in ['country', 'age', 'gender', 'ethnicity']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    joblib.dump(encoders, "multitask_encoders.pkl")
    print("Label encoders updated and saved successfully.")

    # Train/Validation Split (80/20)
    train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    val_df = df.drop(train_df.index).reset_index(drop=True)
    
    # Compute Weights for the Sampler
    print("Computing sample weights for training balance...")
    train_country_counts = train_df['country_encoded'].value_counts().to_dict()
    class_weights = {cls: 1.0 / count for cls, count in train_country_counts.items()}
    
    sample_weights = train_df['country_encoded'].map(class_weights).values
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Transforms (Data Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MultiTaskDataset(train_df, transform=train_transform)
    val_dataset = MultiTaskDataset(val_df, transform=val_transform)
    
    # DataLoaders (num_workers=0 to protect 8GB unified memory on M1)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Initialize Model Configuration
    model = MultiTaskMacClassifier(
        num_countries=len(encoders['country'].classes_),
        num_ages=len(encoders['age'].classes_),
        num_genders=len(encoders['gender'].classes_),
        num_ethnicities=len(encoders['ethnicity'].classes_),
        lr=3e-4
    )
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True),
        ModelCheckpoint(monitor="val_loss", filename="best_region_model", save_top_k=1, mode="min")
    ]
    
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="mps",
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=10
    )
    
    print("\n🚀 Commencing Macro-Region Training Execution Loop...")
    trainer.fit(model, train_loader, val_loader)
    
    trainer.model.to("cpu")
    torch.save(model.state_dict(), "multitask_model_final.ckpt")
    print("✨ Macro-Region training lifecycle finished safely.")

if __name__ == "__main__":
    run_balanced_training()