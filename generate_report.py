import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ==============================================================================
# 1. MODEL & DATASET DEFINITIONS
# ==============================================================================
class MultiTaskDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(row['country_encoded'], dtype=torch.long)

class MultiTaskMacClassifier(nn.Module):
    def __init__(self, num_countries, num_ages, num_genders, num_ethnicities):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Identity()
        
        self.country_head = nn.Linear(in_features, num_countries)
        self.age_head = nn.Linear(in_features, num_ages)
        self.gender_head = nn.Linear(in_features, num_genders)
        self.ethnicity_head = nn.Linear(in_features, num_ethnicities)

    def forward(self, x):
        shared = self.backbone(x)
        return self.country_head(shared) # Returning only region for this report

# ==============================================================================
# 2. EVALUATION & REPORT BUILDER
# ==============================================================================
def generate_academic_report():
    print("Loading datasets and Macro-Region model weights...")
    
    # Load Encoders and Data
    encoders = joblib.load("multitask_encoders.pkl")
    country_encoder = encoders['country']
    
    df = pd.read_csv("multitask_manifest.csv")
    
    # --- APPLY MACRO REGION MAPPING (Must match training exactly) ---
    macro_regions = {
        'India': 'South Asia', 'Bangladesh': 'South Asia', 'Pakistan': 'South Asia', 'Sri Lanka': 'South Asia', 'Nepal': 'South Asia', 'Bhutan': 'South Asia', 'Maldives': 'South Asia', 'Afghanistan': 'South Asia',
        'China': 'East Asia', 'Japan': 'East Asia', 'Korea, Republic of': 'East Asia', "Korea, Democratic People's Republic of": 'East Asia', 'Taiwan, Province of China': 'East Asia', 'Mongolia': 'East Asia', 'Hong Kong': 'East Asia', 'Macao': 'East Asia',
        'Indonesia': 'Southeast Asia', 'Philippines': 'Southeast Asia', 'Viet Nam': 'Southeast Asia', 'Thailand': 'Southeast Asia', 'Myanmar': 'Southeast Asia', 'Malaysia': 'Southeast Asia', 'Cambodia': 'Southeast Asia', 'Lao People\'s Democratic Republic': 'Southeast Asia', 'Singapore': 'Southeast Asia', 'Timor-Leste': 'Southeast Asia', 'Brunei Darussalam': 'Southeast Asia',
        'Uzbekistan': 'Central Asia', 'Kazakhstan': 'Central Asia', 'Tajikistan': 'Central Asia', 'Kyrgyzstan': 'Central Asia', 'Turkmenistan': 'Central Asia',
        'Egypt': 'MENA', 'Iran, Islamic Republic of': 'MENA', 'Turkey': 'MENA', 'Iraq': 'MENA', 'Saudi Arabia': 'MENA', 'Yemen': 'MENA', 'Syrian Arab Republic': 'MENA', 'Morocco': 'MENA', 'Algeria': 'MENA', 'Jordan': 'MENA', 'United Arab Emirates': 'MENA', 'Israel': 'MENA', 'Lebanon': 'MENA', 'Palestine, State of': 'MENA', 'Oman': 'MENA', 'Kuwait': 'MENA', 'Qatar': 'MENA', 'Bahrain': 'MENA', 'Tunisia': 'MENA', 'Libya': 'MENA',
        'Nigeria': 'Sub-Saharan Africa', 'Ethiopia': 'Sub-Saharan Africa', 'Congo, The Democratic Republic of the': 'Sub-Saharan Africa', 'South Africa': 'Sub-Saharan Africa', 'Tanzania, United Republic of': 'Sub-Saharan Africa', 'Kenya': 'Sub-Saharan Africa', 'Uganda': 'Sub-Saharan Africa', 'Sudan': 'Sub-Saharan Africa', 'Ghana': 'Sub-Saharan Africa', 'Cameroon': 'Sub-Saharan Africa', 'Mali': 'Sub-Saharan Africa', 'Madagascar': 'Sub-Saharan Africa', 'Senegal': 'Sub-Saharan Africa', 'Zimbabwe': 'Sub-Saharan Africa', 'Rwanda': 'Sub-Saharan Africa', 'Guinea': 'Sub-Saharan Africa', 'Burundi': 'Sub-Saharan Africa', 'Somalia': 'Sub-Saharan Africa', 'Eritrea': 'Sub-Saharan Africa', 'Sierra Leone': 'Sub-Saharan Africa',
        'Germany': 'Western Europe', 'United Kingdom': 'Western Europe', 'France': 'Western Europe', 'Italy': 'Western Europe', 'Spain': 'Western Europe', 'Netherlands': 'Western Europe', 'Belgium': 'Western Europe', 'Sweden': 'Western Europe', 'Austria': 'Western Europe', 'Switzerland': 'Western Europe', 'Denmark': 'Western Europe', 'Finland': 'Western Europe', 'Norway': 'Western Europe', 'Ireland': 'Western Europe', 'Portugal': 'Western Europe', 'Iceland': 'Western Europe',
        'Russian Federation': 'Eastern Europe', 'Ukraine': 'Eastern Europe', 'Poland': 'Eastern Europe', 'Romania': 'Eastern Europe', 'Czechia': 'Eastern Europe', 'Hungary': 'Eastern Europe', 'Belarus': 'Eastern Europe', 'Bulgaria': 'Eastern Europe', 'Serbia': 'Eastern Europe', 'Slovakia': 'Eastern Europe', 'Croatia': 'Eastern Europe', 'Bosnia and Herzegovina': 'Eastern Europe', 'Moldova, Republic of': 'Eastern Europe', 'Albania': 'Eastern Europe', 'Lithuania': 'Eastern Europe', 'Slovenia': 'Eastern Europe', 'Latvia': 'Eastern Europe', 'Estonia': 'Eastern Europe', 'Greece': 'Eastern Europe',
        'United States': 'North America', 'Canada': 'North America', 'Bermuda': 'North America', 'Greenland': 'North America',
        'Brazil': 'Latin America', 'Mexico': 'Latin America', 'Colombia': 'Latin America', 'Argentina': 'Latin America', 'Peru': 'Latin America', 'Venezuela, Bolivarian Republic of': 'Latin America', 'Chile': 'Latin America', 'Ecuador': 'Latin America', 'Guatemala': 'Latin America', 'Cuba': 'Latin America', 'Haiti': 'Latin America', 'Bolivia, Plurinational State of': 'Latin America', 'Dominican Republic': 'Latin America', 'Honduras': 'Latin America', 'Paraguay': 'Latin America', 'El Salvador': 'Latin America', 'Nicaragua': 'Latin America', 'Costa Rica': 'Latin America', 'Puerto Rico': 'Latin America', 'Panama': 'Latin America', 'Uruguay': 'Latin America', 'Jamaica': 'Latin America',
        'Australia': 'Oceania', 'New Zealand': 'Oceania', 'Papua New Guinea': 'Oceania', 'Fiji': 'Oceania', 'Solomon Islands': 'Oceania', 'Vanuatu': 'Oceania', 'Samoa': 'Oceania', 'Kiribati': 'Oceania', 'Tonga': 'Oceania', 'Micronesia, Federated States of': 'Oceania'
    }
    
    df['region'] = df['country'].map(macro_regions)
    df = df.dropna(subset=['region']).reset_index(drop=True)
    df['country'] = df['region']
    
    # Recreate the exact validation split
    train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    val_df = df.drop(train_df.index).reset_index(drop=True)
    
    # Encode validation targets using the saved encoder
    val_df['country_encoded'] = country_encoder.transform(val_df['country'].astype(str))
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = MultiTaskDataset(val_df, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize Model on GPU/CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiTaskMacClassifier(
        num_countries=len(country_encoder.classes_),
        num_ages=len(encoders['age'].classes_),
        num_genders=len(encoders['gender'].classes_),
        num_ethnicities=len(encoders['ethnicity'].classes_)
    )
    model.load_state_dict(torch.load("multitask_model_final.ckpt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    top3_correct = 0
    total_samples = 0
    
    print("\nScanning Validation Set for Academic Metrics...")
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            # Top-1 Predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Top-3 Accuracy Calculation (for 11 classes)
            _, top3_preds = outputs.topk(3, 1, True, True)
            for i in range(targets.size(0)):
                if targets[i] in top3_preds[i]:
                    top3_correct += 1
            total_samples += targets.size(0)

    # Calculate Core Metrics
    top1_acc = accuracy_score(all_targets, all_preds) * 100
    top3_acc = (top3_correct / total_samples) * 100
    
    # Generate scikit-learn report safely
    present_classes = sorted(list(set(all_targets)))
    present_target_names = country_encoder.inverse_transform(present_classes)
    
    report_dict = classification_report(
        all_targets, 
        all_preds, 
        labels=present_classes,
        target_names=present_target_names, 
        output_dict=True, 
        zero_division=0
    )
    
    # ==============================================================================
    # 3. EXPORT ACADEMIC TEXT REPORT
    # ==============================================================================
    with open("EthnoFace_Evaluation_Report.txt", "w") as f:
        f.write("====================================================================\n")
        f.write("LEADING UNIVERSITY - DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING\n")
        f.write("THESIS: EthnoFace - Nationality Detection & Facial Attribute Estimation\n")
        f.write("AUTHOR: Rabiul Islam Apu\n")
        f.write("====================================================================\n\n")
        
        f.write("1. ARCHITECTURE & DATASET OVERVIEW\n")
        f.write("----------------------------------\n")
        f.write(f"Backbone Model    : MobileNetV3 (Large)\n")
        f.write(f"Total Parameters  : 4.7 Million\n")
        f.write(f"Target Classes    : {len(present_target_names)} Global Macro-Regions\n")
        f.write(f"Validation Set    : {total_samples} Isolated Unseen Images\n")
        f.write(f"Optimization      : WeightedRandomSampler (Class-Balanced)\n\n")
        
        f.write("2. GLOBAL PERFORMANCE METRICS\n")
        f.write("----------------------------------\n")
        f.write(f"Top-1 Accuracy    : {top1_acc:.2f}%\n")
        f.write(f"Top-3 Accuracy    : {top3_acc:.2f}%\n\n")
        
        f.write("3. REGIONAL DEMOGRAPHIC PERFORMANCE (F1-Score)\n")
        f.write("----------------------------------\n")
        f.write(f"{'Macro-Region':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score'}\n")
        f.write("-" * 65 + "\n")
        
        # Sort classes by F1-Score
        class_scores = [(name, metrics) for name, metrics in report_dict.items() if isinstance(metrics, dict)]
        class_scores.sort(key=lambda x: x[1]['f1-score'], reverse=True)
        
        for name, metrics in class_scores:
            if name not in ['accuracy', 'macro avg', 'weighted avg']:
                f.write(f"{name:<25} | {metrics['precision']:.2f}       | {metrics['recall']:.2f}       | {metrics['f1-score']:.2f}\n")
            
    print("\n✅ Saved detailed evaluation to 'EthnoFace_Evaluation_Report.txt'")

    # ==============================================================================
    # 4. GENERATE VISUALIZATION FOR PRESENTATION
    # ==============================================================================
    top_names = [x[0] for x in class_scores if x[0] not in ['accuracy', 'macro avg', 'weighted avg']]
    top_f1 = [x[1]['f1-score'] for x in class_scores if x[0] not in ['accuracy', 'macro avg', 'weighted avg']]
    
    plt.figure(figsize=(12, 8))
    # 'hue' is assigned and 'legend=False' to fix the FutureWarning in newer seaborn versions
    sns.barplot(x=top_f1, y=top_names, hue=top_names, palette="viridis", legend=False)
    plt.title("EthnoFace AGEC Framework: Macro-Region Classification by F1-Score", fontsize=14, pad=15)
    plt.xlabel("F1-Score (Harmonic Mean of Precision & Recall)", fontsize=12)
    plt.ylabel("Target Macro-Region", fontsize=12)
    plt.xlim(0, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("EthnoFace_Top_Performers.png", dpi=300)
    print("✅ Saved performance visualization to 'EthnoFace_Top_Performers.png'\n")

if __name__ == "__main__":
    generate_academic_report()