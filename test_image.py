import torch
import joblib
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
import torch.nn as nn

# 1. Model definition (Automatically adjusts to the 11 regions via the encoder)
class MultiTaskMacClassifier(pl.LightningModule):
    def __init__(self, num_countries, num_ages, num_genders, num_ethnicities, lr=3e-4):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=None) 
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Identity()
        
        # 'country_head' now predicts the 11 Macro-Regions
        self.country_head = nn.Linear(in_features, num_countries)
        self.age_head = nn.Linear(in_features, num_ages)
        self.gender_head = nn.Linear(in_features, num_genders)
        self.ethnicity_head = nn.Linear(in_features, num_ethnicities)

    def forward(self, x):
        shared = self.backbone(x)
        return {
            'country': self.country_head(shared),
            'age': self.age_head(shared),
            'gender': self.gender_head(shared),
            'ethnicity': self.ethnicity_head(shared)
        }

def predict_image(image_path, model_path="multitask_model_final.ckpt", encoders_path="multitask_encoders.pkl"):
    print(f"Loading image: {image_path}...")
    
    encoders = joblib.load(encoders_path)
    
    model = MultiTaskMacClassifier(
        num_countries=len(encoders['country'].classes_),
        num_ages=len(encoders['age'].classes_),
        num_genders=len(encoders['gender'].classes_),
        num_ethnicities=len(encoders['ethnicity'].classes_)
    )
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval() 
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)
        
    print("\n" + "="*50)
    print("🤖 ETHNOFACE PREDICTION RESULTS")
    print("="*50)
    
    # Print Age, Gender, Ethnicity
    for category in ['age', 'gender', 'ethnicity']:
        pred_idx = torch.argmax(predictions[category], dim=1).item()
        pred_label = encoders[category].inverse_transform([pred_idx])[0]
        confidence = F.softmax(predictions[category], dim=1)[0][pred_idx].item() * 100
        print(f" {category.capitalize():<10}: {pred_label:<15} ({confidence:.1f}% confidence)")
        
    print("-" * 50)
    print(" 🌍 MACRO-REGION PROBABILITY DISTRIBUTION")
    print("-" * 50)
    
    # Calculate probabilities for the 11 Macro-Regions
    region_probs = F.softmax(predictions['country'], dim=1)[0]
    
    # Get the Top 3 highest scoring regions
    top3_probs, top3_indices = torch.topk(region_probs, 3)
    
    for i in range(3):
        prob_val = top3_probs[i].item() * 100
        region_name = encoders['country'].inverse_transform([top3_indices[i].item()])[0]
        
        bar_length = int(prob_val / 4) # Scaled for terminal fitting
        bar = "█" * bar_length
        
        print(f" {i+1}. {region_name:<20}: {prob_val:>5.1f}% | {bar}")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_picture = "test.jpg" 
    
    try:
        predict_image(test_picture)
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find '{test_picture}'. Please put a photo in the folder and name it 'test.jpg'!")