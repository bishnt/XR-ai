import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
import sys

# Add backend to path so we can import our model
sys.path.append(os.getcwd())
from model import load_model

def debug_prediction(image_path):
    device = 'cpu'
    model = load_model(model_path='best_xray_model.pth', device=device)
    model.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    with open('backend/debug_output.txt', 'w') as f:
        f.write(f"--- Results for {os.path.basename(image_path)} ---\n")
        current_classes = ['Normal', 'Pneumonia', 'COVID', 'TB']
        for i, (name, prob) in enumerate(zip(current_classes, probabilities)):
            f.write(f"Index {i} ({name}): {prob.item():.4f}\n")

        predicted_idx = torch.argmax(probabilities).item()
        f.write(f"\nPredicted Index: {predicted_idx}\n")
        f.write(f"Predicted Class in current code: {current_classes[predicted_idx]}\n")
    
    # Also print to terminal
    print(f"Results written to backend/debug_output.txt")

if __name__ == "__main__":
    # Check if there's a file in uploads
    # Try both local and backend/uploads
    uploads_dir = 'backend/uploads' if os.path.exists('backend/uploads') else 'uploads'
    if os.path.exists(uploads_dir):
        files = [f for f in os.listdir(uploads_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            debug_prediction(os.path.join(uploads_dir, files[0]))
        else:
            print(f"No images found in {uploads_dir} for testing.")
    else:
        print(f"Neither 'uploads' nor 'backend/uploads' directory found (CWD: {os.getcwd()})")
