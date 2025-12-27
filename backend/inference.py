"""
Inference Engine for X-Ray Image Classification

Handles image preprocessing, model inference, and result post-processing
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import time


from gradcam import GradCAM

class XRayInference:
    """
    Inference engine for X-Ray image classification
    
    Handles the complete pipeline:
    1. Image loading and preprocessing
    2. Model inference
    3. Result post-processing
    4. Grad-CAM generation
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize inference engine
        
        Args:
            model: Trained XRayClassifier model
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.eval()  # Set to evaluation mode
        
        # Initialize Grad-CAM
        self.gradcam = GradCAM(model)
        
        # Image preprocessing transform
        # Following ImageNet normalization standards
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # EfficientNet-B0 input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]     # ImageNet std
            )
        ])
        
        # Class names
        self.class_names = ['Normal', 'Pneumonia', 'COVID', 'TB']
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess image for model input
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Lowiad image
            image = Image.open(image_path)
            
            # Convert to RGB (in case of grayscale or RGBA)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor
        
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    def predict(self, image_path):
        """
        Run inference on an image
        
        Args:
            image_path (str): Path to X-ray image
        
        Returns:
            dict: Prediction results containing:
                - class: Predicted class name
                - confidence: Confidence score (0-1)
                - probabilities: Dict of all class probabilities
                - processing_time_ms: Inference time in milliseconds
                - heatmap: Base64 encoded heatmap overlay string
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(image_tensor)
                probabilities = F.softmax(logits, dim=1)
            
            # Get prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Get all class probabilities
            all_probs = probabilities[0].cpu().numpy()
            prob_dict = {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, all_probs)
            }
            
            # Generate Grad-CAM Heatmap
            # We want to explain the predicted class
            # Note: We need to enable grad for the heatmap generation even in inference
            # But the model parameters are frozen, so we just need grad for activations
            heatmap = self.gradcam.generate(image_tensor, class_idx=predicted_idx.item())
            
            # Create overlay
            heatmap_overlay = self.gradcam.get_base64_overlay(heatmap, image_path)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'class': predicted_class,
                'confidence': float(confidence_score),
                'probabilities': prob_dict,
                'processing_time_ms': round(processing_time, 2),
                'heatmap': heatmap_overlay
            }
        
        except Exception as e:
            raise RuntimeError(f"Inference error: {str(e)}")
    
    def predict_batch(self, image_paths):
        """
        Run inference on multiple images
        
        Args:
            image_paths (list): List of image paths
        
        Returns:
            list: List of prediction results (one per image)
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'image_path': image_path
                })
        
        return results
    
    def get_top_predictions(self, image_path, top_k=2):
        """
        Get top-k predictions for an image
        
        Args:
            image_path (str): Path to image
            top_k (int): Number of top predictions to return
        
        Returns:
            list: List of (class_name, confidence) tuples
        """
        result = self.predict(image_path)
        probs = result['probabilities']
        
        # Sort by probability
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_probs[:top_k]


def create_inference_engine(model, device='cpu'):
    """
    Factory function to create inference engine
    
    Args:
        model: Trained XRayClassifier model
        device (str): Device to use ('cpu' or 'cuda')
    
    Returns:
        XRayInference: Inference engine instance
    """
    return XRayInference(model, device)


if __name__ == "__main__":
    # Test inference engine
    print("Testing X-Ray Inference Engine...")
    
    from model import load_model
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = load_model(device=device)
    
    # Create inference engine
    inference = create_inference_engine(model, device)
    print("✓ Inference engine created successfully")
    print(f"  Classes: {inference.class_names}")
    
    # Test with dummy image (if exists)
    # Replace with actual test image path
    # result = inference.predict('test_xray.jpg')
    # print(f"✓ Prediction: {result}")
