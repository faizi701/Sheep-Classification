import torch
import torchvision.transforms as transforms
from PIL import Image
from model_architecture import SheepCNN
import sys
import os

# ---------- Configuration ----------
MODEL_PATH = 'models/sheep_cnn.pth'  
CLASS_NAMES = ['Barbari', 'Goat', 'Harri', 'Naeimi', 'Najdi', 'Roman', 'Sawakni']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Load Model ----------
model = SheepCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------- Image Transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error opening image: {image_path} - {e}")
        return None

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]

# ---------- CLI Support ----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️  Usage: python predict.py <image_path1> <image_path2> ...")
        sys.exit()

    image_paths = sys.argv[1:]
    for path in image_paths:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue
        pred = predict_image(path)
        print(f"📷 Image: {path} → 🐑 Predicted: {pred}")
