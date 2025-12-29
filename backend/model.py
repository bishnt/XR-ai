"""
X-Ray Image Classification Model
Based on EfficientNet-B0 architecture from ku.ipynb (3rd block)

This module contains the neural network architecture for classifying
chest X-ray images into 4 categories: NORMAL, PNEUMONIA, COVID, TB
"""

import torch
import torch.nn as nn
from torchvision import models


class XRayClassifier(nn.Module):
    """
    X-Ray Image Classifier using EfficientNet-B0 backbone
    
    Architecture:
    - EfficientNet-B0 pretrained on ImageNet
    - Custom classification head with dropout
    - 4 output classes: NORMAL, PNEUMONIA, COVID, TB
    """
    
    def __init__(self, num_classes=4, pretrained=True):
        """
        Initialize the X-Ray Classifier
        
        Args:
            num_classes (int): Number of output classes (default: 4)
            pretrained (bool): Use ImageNet pretrained weights (default: True)
        """
        super(XRayClassifier, self).__init__()
        
        # Load EfficientNet-B0 backbone
        # Better than ResNet50 for medical images due to compound scaling
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Get the number of input features to the classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier head with custom layers
        # Dropout helps prevent overfitting on medical images
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(512, num_classes)
        )
        
        # Class names mapping (Alphabetical: COVID, Normal, Pneumonia, TB)
        self.class_names = ['COVID', 'Normal', 'Pneumonia', 'TB']
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_class_name(self, class_idx):
        """
        Get class name from index
        
        Args:
            class_idx (int): Class index (0-3)
        
        Returns:
            str: Class name
        """
        return self.class_names[class_idx]


def load_model(model_path=None, device='cpu'):
    """
    Load the X-Ray classifier model
    
    Args:
        model_path (str): Path to saved model weights (optional)
        device (str): Device to load model on ('cpu' or 'cuda')
    
    Returns:
        XRayClassifier: Loaded model in evaluation mode
    """
    # Initialize model
    model = XRayClassifier(num_classes=4, pretrained=True)
    
    # Load saved weights if provided
    if model_path:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different saving formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Loaded a full checkpoint (like from the notebook)
                state_dict = checkpoint['model_state_dict']
                # Remove 'module.' prefix if saved from DataParallel, though unlikely here
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict
            else:
                # Loaded direct state dict
                state_dict = checkpoint
                
            model.load_state_dict(state_dict)
            print(f"✓ Model weights loaded from {model_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load model weights: {e}")
            print("  Using pretrained EfficientNet-B0 backbone with random classifier (Results will be inaccurate)")
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test model instantiation
    print("Testing X-Ray Classifier model...")
    
    # Create model
    model = XRayClassifier(num_classes=4, pretrained=False)
    print(f"✓ Model created successfully")
    print(f"  Classes: {model.class_names}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
