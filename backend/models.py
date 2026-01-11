"""
Model loading and inference utilities.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from collections import OrderedDict

from backend.config import (
    MODELS_DIR, NUM_CLASSES,
    SUPPORTED_MODELS, DEFAULT_MODEL
)
from backend.labels_22 import DIALECT_LABELS


class SpectralCNN(nn.Module):
    """
    Spectral CNN architecture for audio classification.
    
    IMPORTANT: This architecture MUST match the training SCNN exactly.
    If SCNN predictions are wrong (e.g., always predicting Djibouti with low confidence),
    verify this matches adc/models/scnn.py or the SCNN definition in training notebooks.
    
    Expected input: (batch, 1, 128, T) where:
    - 1 = single mel channel
    - 128 = n_mels
    - T = number of time frames
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Input: (batch, 1, 128, time_frames)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def create_model(architecture: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Create a model instance based on architecture name.
    
    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    if architecture == "resnet18":
        model = models.resnet18(pretrained=False)
        # Modify first layer: average RGB weights to 1 channel (as in training)
        old_conv1 = model.conv1
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Average the RGB weights: (64, 3, 7, 7) -> (64, 1, 7, 7)
        with torch.no_grad():
            new_conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)
        model.conv1 = new_conv1
        # Modify last layer for num_classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif architecture == "resnet50":
        model = models.resnet50(pretrained=False)
        # Modify first layer: average RGB weights to 1 channel (as in training)
        old_conv1 = model.conv1
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Average the RGB weights
        with torch.no_grad():
            new_conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)
        model.conv1 = new_conv1
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif architecture == "densenet121":
        model = models.densenet121(pretrained=False)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif architecture == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif architecture == "efficientnet_b3":
        try:
            # Try new torchvision API first
            try:
                from torchvision.models import efficientnet_b3
                model = efficientnet_b3(pretrained=False)
            except ImportError:
                # Fallback to older API
                from torchvision.models import efficientnet
                model = efficientnet.efficientnet_b3(pretrained=False)
            
            # Modify first layer for single channel input
            if hasattr(model, 'features'):
                model.features[0][0] = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
            elif hasattr(model, 'conv_stem'):
                model.conv_stem = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
            
            # Modify classifier
            if hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
                else:
                    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        except Exception as e:
            raise ImportError(f"Failed to load efficientnet_b3: {e}. Requires torchvision >= 0.13.0")
            
    elif architecture == "scnn":
        model = SpectralCNN(num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def find_latest_model_dir(model_name: str) -> Optional[Path]:
    """
    Find the latest timestamped directory for a model.
    Supports both YYYYMMDD_HHMMSS and YYYYMMDD_* patterns.
    
    Args:
        model_name: Model architecture name
        
    Returns:
        Path to latest model directory, or None if not found
    """
    model_base_dir = MODELS_DIR / model_name
    if not model_base_dir.exists():
        return None
    
    # Find all timestamp directories
    # Pattern: YYYYMMDD_HHMMSS or YYYYMMDD_*
    timestamp_dirs = []
    for d in model_base_dir.iterdir():
        if d.is_dir():
            # Check if it matches timestamp pattern (8 digits, underscore, 6 digits or wildcard)
            parts = d.name.split('_')
            if len(parts) == 2 and len(parts[0]) == 8 and parts[0].isdigit():
                timestamp_dirs.append(d)
    
    if not timestamp_dirs:
        return None
    
    # Sort by name (timestamp) and return latest
    timestamp_dirs.sort(key=lambda x: x.name, reverse=True)
    return timestamp_dirs[0]


def load_model_checkpoint(model_name: str, device: torch.device) -> nn.Module:
    """
    Load a model from checkpoint.
    
    Args:
        model_name: Model architecture name
        device: Torch device
        
    Returns:
        Loaded model in eval mode
    """
    model_dir = find_latest_model_dir(model_name)
    if model_dir is None:
        raise FileNotFoundError(f"Model directory not found for {model_name}")
    
    checkpoint_path = model_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint first to inspect structure
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # Check first layer input channels for EfficientNet
    # If checkpoint has 3-channel input, we need to handle it differently
    first_layer_key = None
    if model_name == "efficientnet_b3":
        # Find the first conv layer key
        for key in new_state_dict.keys():
            if 'features.0.0.weight' in key or 'conv_stem.weight' in key:
                first_layer_key = key
                break
        
        if first_layer_key and new_state_dict[first_layer_key].shape[1] == 3:
            # Checkpoint expects 3-channel input, create model without modifying first layer
            try:
                from torchvision.models import efficientnet_b3
                model = efficientnet_b3(pretrained=False)
            except ImportError:
                from torchvision.models import efficientnet
                model = efficientnet.efficientnet_b3(pretrained=False)
            
            # Only modify classifier, not first layer
            if hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
                else:
                    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        else:
            # Create model with 1-channel input as normal
            model = create_model(model_name, num_classes=NUM_CLASSES)
    else:
        # Create model normally for other architectures
        model = create_model(model_name, num_classes=NUM_CLASSES)
    
    # Load state dict with strict checking for SCNN, lenient for others
    if model_name == "scnn":
        # For SCNN, we want strict loading to catch architecture mismatches
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"⚠ SCNN missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"⚠ SCNN unexpected keys: {unexpected_keys}")
        
        # Try strict loading to verify architecture matches
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print(f"✓ SCNN loaded with strict=True (architecture matches)")
        except RuntimeError as e:
            print(f"✗ SCNN strict loading failed: {e}")
            print("  This indicates an architecture mismatch!")
            print("  Check that SCNN definition matches training code exactly.")
            # Continue with non-strict for now, but this is a problem
    else:
        # For other models, use lenient loading
        try:
            model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Some weights could not be loaded: {e}")
            # Filter out mismatched keys
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in new_state_dict.items() 
                            if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
    
    model.to(device)
    model.eval()
    
    return model


def adapt_input_for_efficientnet(tensor: torch.Tensor) -> torch.Tensor:
    """
    Adapt input for EfficientNet-B3:
    - 3-channel input (repeat 1->3)
    - Resize to 300x300
    
    Args:
        tensor: Input tensor, shape (B, 1, n_mels, T)
        
    Returns:
        Adapted tensor, shape (B, 3, 300, 300)
    """
    # Repeat channel: (B, 1, n_mels, T) -> (B, 3, n_mels, T)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    
    # Resize to 300x300
    tensor = torch.nn.functional.interpolate(
        tensor, size=(300, 300), mode='bilinear', align_corners=False
    )
    
    return tensor


def adapt_input_for_scnn(tensor: torch.Tensor) -> torch.Tensor:
    """
    Adapt input for SCNN:
    - Fixed time dimension = 700 frames
    - Pad or truncate along time axis
    
    Args:
        tensor: Input tensor, shape (B, 1, 128, T)
        
    Returns:
        Adapted tensor, shape (B, 1, 128, 700)
    """
    target_time = 700
    
    if tensor.shape[3] < target_time:
        # Pad right
        pad = target_time - tensor.shape[3]
        tensor = torch.nn.functional.pad(tensor, (0, pad), mode='constant', value=0)
    elif tensor.shape[3] > target_time:
        # Truncate right
        tensor = tensor[:, :, :, :target_time]
    
    return tensor


def adapt_input_for_default_cnn(tensor: torch.Tensor) -> torch.Tensor:
    """
    Adapt input for default CNNs (ResNet, DenseNet, MobileNet):
    - 1-channel input (already correct)
    - Variable time axis allowed
    
    Args:
        tensor: Input tensor, shape (B, 1, n_mels, T)
        
    Returns:
        Unchanged tensor
    """
    return tensor


class DialectModel:
    """
    Wrapper class for dialect identification models.
    """
    def __init__(self, name: str, architecture: str, device: torch.device):
        self.name = name
        self.architecture = architecture
        self.device = device
        self.model = load_model_checkpoint(architecture, device)
        self.num_classes = NUM_CLASSES
        
        # Check if model expects 3-channel input (for EfficientNet)
        self.expects_3channel = False
        if architecture == "efficientnet_b3":
            # Check first layer input channels
            if hasattr(self.model, 'features') and len(self.model.features) > 0:
                first_conv = self.model.features[0]
                if hasattr(first_conv, '0') and hasattr(first_conv[0], 'in_channels'):
                    self.expects_3channel = first_conv[0].in_channels == 3
            elif hasattr(self.model, 'conv_stem'):
                self.expects_3channel = self.model.conv_stem.in_channels == 3
        
    def predict_from_tensor(
        self,
        model_tensor: torch.Tensor,
        window_mode: str
    ) -> Dict:
        """
        Run inference on preprocessed model tensor.
        
        Args:
            model_tensor: Preprocessed tensor
                - For 7s/3s_center: shape (1, 1, n_mels, T)
                - For 3s_5crop: shape (5, 1, n_mels, T3)
            window_mode: Window mode used
            
        Returns:
            Dictionary with predictions and probabilities
        """
        with torch.inference_mode():
            # Apply model-specific input adapter
            if self.architecture == "efficientnet_b3":
                model_tensor = adapt_input_for_efficientnet(model_tensor)
            elif self.architecture == "scnn":
                model_tensor = adapt_input_for_scnn(model_tensor)
            else:
                model_tensor = adapt_input_for_default_cnn(model_tensor)
            
            model_tensor = model_tensor.to(self.device)
            
            if window_mode == '3s_5crop':
                # Handle 5-crop: model_tensor shape is (5, 1, n_mels, T3) or adapted
                # Forward pass on batch of 5
                logits_5 = self.model(model_tensor)  # Shape: (5, num_classes)
                
                # Average probabilities (not logits) as per training evaluation
                probs_5 = torch.softmax(logits_5, dim=1)  # (5, num_classes)
                probs_avg = probs_5.mean(dim=0).cpu().numpy()  # (num_classes,)
            else:
                # Single prediction: model_tensor shape is (1, 1, n_mels, T) or adapted
                # Forward pass
                logits = self.model(model_tensor)  # Shape: (1, num_classes)
                probs_avg = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (num_classes,)
            
            # Get top prediction
            top_idx = int(np.argmax(probs_avg))
            dialect = DIALECT_LABELS[top_idx]
            confidence = float(probs_avg[top_idx])
            
            # Get top-k (top 5)
            top_k_indices = np.argsort(probs_avg)[::-1][:5]
            top_k = [
                {
                    "dialect": DIALECT_LABELS[idx],
                    "prob": float(probs_avg[idx])
                }
                for idx in top_k_indices
            ]
            
            # All probabilities
            all_probs = {
                DIALECT_LABELS[i]: float(probs_avg[i])
                for i in range(len(DIALECT_LABELS))
            }
            
            return {
                "dialect": dialect,
                "confidence": confidence,
                "top_k": top_k,
                "all_probs": all_probs
            }


class ModelRegistry:
    """
    Registry for managing all loaded models.
    """
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.models: Dict[str, DialectModel] = {}
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all available models."""
        for model_name in SUPPORTED_MODELS:
            try:
                model = DialectModel(model_name, model_name, self.device)
                self.models[model_name] = model
                print(f"✓ Loaded model: {model_name}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
    
    def get_model(self, model_name: str) -> DialectModel:
        """
        Get a model by name.
        
        Args:
            model_name: Model name or 'best' for default
            
        Returns:
            DialectModel instance
        """
        if model_name == "best" or model_name == "Best (recommended)":
            model_name = DEFAULT_MODEL
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        return self.models[model_name]
    
    def list_models(self) -> List[Dict]:
        """
        List all available models with metadata.
        
        Returns:
            List of model metadata dictionaries
        """
        return [
            {
                "name": name,
                "architecture": model.architecture,
                "num_classes": model.num_classes
            }
            for name, model in self.models.items()
        ]

