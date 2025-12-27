
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.cm as cm
from PIL import Image
import io
import base64

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
    for visualizing model decisions.
    
    Targeted for EfficientNet-B0 backbone.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: The PyTorch model
            target_layer: The specific layer to hook into. 
                          If None, defaults to the last convolutional layer of EfficientNet-B0.
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # Determine target layer (last block of EfficientNet features)
        if target_layer is None:
            # For efficientnet_b0, features[-1] is the last convolutional block before pooling
            # Accessing via the backbone attribute we set in model.py
            self.target_layer = self.model.backbone.features[-1]
        else:
            self.target_layer = target_layer
            
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple, access the first element
        self.gradients = grad_output[0]
        
    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W)
            class_idx: Index of the class to visualize. If None, uses the predicted class.
            
        Returns:
            np.array: Heatmap (H, W) normalized to 0-1
        """
        # Zero gradients
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Backward pass
        # Target score for the specific class
        target = output[0][class_idx]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        # pooled_gradients: [batch_size, channels, 1, 1] -> [batch_size, channels]
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight activations by the gradients
        # activations: [batch_size, channels, H, W]
        # We process single image batch here
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the weighted activations
        # heatmap: [batch_size, H, W] -> [H, W] (squeeze batch dim)
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU extraction (only positive influence)
        heatmap = F.relu(heatmap)
        
        # Normalize
        heatmap = heatmap.detach().cpu().numpy()
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap

    def overlay_heatmap(self, heatmap, original_image_path, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Normalized heatmap (0-1)
            original_image_path: Path to the original image file
            alpha: Opacity of the heatmap (0-1)
            colormap: OpenCV colormap constant
            
        Returns:
            PIL.Image: Image with heatmap overlay
        """
        # Load original image
        img = cv2.imread(original_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to RGB using colormap
        # Scale to 0-255 and convert to uint8
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        # Note: applyColorMap returns BGR, so we need to convert to RGB if we want consistency,
        # but since we are usually working with BGR in OpenCV, let's keep it consistent locally before merging.
        # Actually standard JET map in opencv is BGR.
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend images
        superimposed = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)
        
        return Image.fromarray(superimposed)
        
    def get_base64_overlay(self, heatmap, original_image_path):
        """
        Get base64 encoded string of the overlay image
        
        Args:
            heatmap: Normalized heatmap
            original_image_path: Path to original image
            
        Returns:
            str: Base64 data string
        """
        overlay_img = self.overlay_heatmap(heatmap, original_image_path)
        
        # Save to buffer
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="PNG")
        
        # Encode
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
