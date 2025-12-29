import torch
import os
import sys

# Ensure backend folder is in path
sys.path.append(os.path.join(os.getcwd(), 'backend'))
try:
    from model import XRayClassifier
except ImportError:
    from backend.model import XRayClassifier

def check_weights(model_path):
    model = XRayClassifier(num_classes=4)
    try:
        if not os.path.exists(model_path):
            return f"Error: File {model_path} not found"
            
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check type of checkpoint
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            state_dict = checkpoint
            
        # Try to load
        model.load_state_dict(state_dict)
        return "SUCCESS"
    except Exception as e:
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
    result = check_weights('backend/best_xray_model.pth')
    print(result)
    with open('backend/load_error.txt', 'w') as f:
        f.write(result)
