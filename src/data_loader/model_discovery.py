"""
Model discovery utilities for finding available models.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional


class ModelDiscovery:
    """Handles discovery of available models and data."""
    
    @staticmethod
    def discover_available_models(data_dir: Path) -> List[str]:
        """
        Discover available models from the data directory structure.
        
        Args:
            data_dir: Base data directory
            
        Returns:
            List of discovered model names
        """
        models_dir = data_dir / "images"
        
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Get all subdirectories in models directory
        model_names = []
        for item in models_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                model_names.append(item.name)
        
        if not model_names:
            raise ValueError(f"No model directories found in {models_dir}")
        
        # Sort models by the same order used in training to ensure label consistency
        try:
            from src.utils.model_isolation import sort_models_by_load_order
            sorted_model_names = sort_models_by_load_order(model_names)
            print(f"📁 Discovered {len(sorted_model_names)} models: {sorted_model_names}")
            return sorted_model_names
        except ImportError:
            # Fallback: sort alphabetically
            sorted_model_names = sorted(model_names)
            print(f"📁 Discovered {len(sorted_model_names)} models (alphabetical): {sorted_model_names}")
            return sorted_model_names
    
    @staticmethod
    def check_data_availability(
        data_dir: Path, 
        fingerprint_method: str, 
        num_images_per_model: int
    ) -> Dict[str, bool]:
        """
        Check what data is available for a given fingerprint method.
        
        Args:
            data_dir: Base data directory
            fingerprint_method: Fingerprint method name
            num_images_per_model: Required number of images per model
            
        Returns:
            Dictionary with availability status
        """
        availability = {
            'images_exist': False,
            'features_exist': False,
            'sufficient_images': False,
            'sufficient_features': False
        }
        
        # Check images
        images_dir = data_dir / "images"
        if images_dir.exists():
            availability['images_exist'] = True
            
            # Check if we have enough images per model
            model_names = ModelDiscovery.discover_available_models(data_dir)
            total_required_images = len(model_names) * num_images_per_model
            
            # Count total images
            total_images = 0
            for model_dir in images_dir.iterdir():
                if model_dir.is_dir():
                    for file in model_dir.iterdir():
                        if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            total_images += 1
            
            availability['sufficient_images'] = total_images >= total_required_images
        
        # Check features
        features_dir = data_dir / f"features_{fingerprint_method}"
        if features_dir.exists():
            availability['features_exist'] = True
            
            # Check if we have enough features per model
            if availability['images_exist']:
                model_names = ModelDiscovery.discover_available_models(data_dir)
                total_required_features = len(model_names) * num_images_per_model
                
                # Count total features
                total_features = 0
                for model_dir in features_dir.iterdir():
                    if model_dir.is_dir():
                        for file in model_dir.iterdir():
                            if file.suffix.lower() in ['.npy', '.pkl', '.pt', '.pth']:
                                total_features += 1
                
                availability['sufficient_features'] = total_features >= total_required_features
        
        return availability
    
    @staticmethod
    def get_data_summary(data_dir: Path, fingerprint_method: str) -> Dict[str, any]:
        """
        Get summary of available data.
        
        Args:
            data_dir: Base data directory
            fingerprint_method: Fingerprint method name
            
        Returns:
            Dictionary with data summary
        """
        summary = {
            'data_dir': str(data_dir),
            'fingerprint_method': fingerprint_method,
            'models': [],
            'images_count': 0,
            'features_count': 0
        }
        
        # Get model information
        try:
            model_names = ModelDiscovery.discover_available_models(data_dir)
            summary['models'] = model_names
            
            # Count images
            images_dir = data_dir / "images"
            if images_dir.exists():
                for model_name in model_names:
                    model_dir = images_dir / model_name
                    if model_dir.exists():
                        image_count = len([f for f in model_dir.iterdir() 
                                         if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
                        summary['images_count'] += image_count
            
            # Count features
            features_dir = data_dir / f"features_{fingerprint_method}"
            if features_dir.exists():
                for model_name in model_names:
                    model_dir = features_dir / model_name
                    if model_dir.exists():
                        feature_count = len([f for f in model_dir.iterdir() 
                                           if f.suffix.lower() in ['.npy', '.pkl', '.pt', '.pth']])
                        summary['features_count'] += feature_count
        
        except (FileNotFoundError, ValueError) as e:
            summary['error'] = str(e)
        
        return summary
