"""
Image saving functionality for attack results.
"""

import torch
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from torchvision import transforms


class ImageSaver:
    """Handles saving of attacked images with proper organization."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def save_attacked_images(
        self,
        original_images: torch.Tensor,
        adversarial_images: torch.Tensor,
        y_labels: torch.Tensor,
        original_predictions: torch.Tensor,
        adversarial_predictions: torch.Tensor,
        attack_type: str,
        attack_goal: str,
        model_names: List[str],
        success_mask=None,
        target_predictions=None
    ) -> None:
        """
        Save successfully attacked images to organized folders.
        
        Args:
            original_images: Original images tensor
            adversarial_images: Adversarial images tensor
            y_labels: True labels (source model labels)
            original_predictions: Original model predictions
            adversarial_predictions: Adversarial model predictions
            attack_type: Type of attack (w1, w2, w3, b1, b2)
            attack_goal: 'removal' or 'forgery'
            model_names: List of model names
            success_mask: Boolean mask indicating successful attacks (optional)
            target_predictions: Target predictions for forgery attacks (optional)
        """
        
        if success_mask is None:
            # Determine success based on attack goal
            if attack_goal == 'removal':
                # Success: originally correct prediction becomes incorrect
                success_mask = (original_predictions == y_labels) & (adversarial_predictions != y_labels)
            elif attack_goal == 'forgery':
                # Success: adversarial prediction matches target
                if target_predictions is not None:
                    success_mask = (adversarial_predictions == target_predictions)
                else:
                    # If no target predictions, use original success logic
                    success_mask = (original_predictions == y_labels) & (adversarial_predictions != y_labels)
            else:
                print(f"Warning: Unknown attack goal '{attack_goal}', skipping image saving")
                return
        
        if not success_mask.any():
            print("No successful attacks found, skipping image saving")
            return
        
        # Create base directory for attacked images
        attacked_images_dir = self.data_dir / "attacked_images"
        attacked_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Get successful indices
        successful_indices = torch.where(success_mask)[0]
        print(f"💾 Saving {len(successful_indices)} successfully attacked images...")
        
        # Create model name mapping
        label_to_model_name = {i: model_name for i, model_name in enumerate(model_names)}
        
        for idx in successful_indices:
            idx = idx.item()
            original_img = original_images[idx]
            adversarial_img = adversarial_images[idx]
            
            # Get source model name
            source_label = y_labels[idx].item()
            source_model = label_to_model_name[source_label]
            
            # Determine folder name based on attack goal
            if attack_goal == 'removal':
                folder_name = source_model
            elif attack_goal == 'forgery':
                target_label = adversarial_predictions[idx].item()
                target_model = label_to_model_name[target_label]
                folder_name = f"{source_model}_{target_model}"
            else:
                folder_name = source_model
            
            # Create attack-specific folder
            attack_folder = attacked_images_dir / attack_type / folder_name
            attack_folder.mkdir(parents=True, exist_ok=True)
            
            # Save original image
            original_filename = f"original_{idx:04d}.png"
            original_path = attack_folder / original_filename
            
            # Convert tensor to PIL Image and save
            original_pil = self._tensor_to_pil(original_img)
            original_pil.save(original_path)
            
            # Save adversarial image
            adversarial_filename = f"adversarial_{idx:04d}.png"
            adversarial_path = attack_folder / adversarial_filename
            
            # Convert tensor to PIL Image and save
            adversarial_pil = self._tensor_to_pil(adversarial_img)
            adversarial_pil.save(adversarial_path)
            
            # Save metadata
            metadata_filename = f"metadata_{idx:04d}.txt"
            metadata_path = attack_folder / metadata_filename
            
            metadata = self._generate_metadata(
                idx, source_label, adversarial_predictions[idx].item(),
                attack_type, attack_goal, original_predictions[idx].item()
            )
            
            with open(metadata_path, 'w') as f:
                f.write(metadata)
        
        print(f"✅ Successfully saved {len(successful_indices)} attacked image pairs to {attacked_images_dir}")
    
    def _tensor_to_pil(self, img_tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        
        # Ensure tensor is in [0, 1] range
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Convert to PIL Image
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(img_tensor)
        
        return pil_image
    
    def _generate_metadata(
        self,
        idx: int,
        source_label: int,
        adversarial_prediction: int,
        attack_type: str,
        attack_goal: str,
        original_prediction: int
    ) -> str:
        """Generate metadata for saved images."""
        
        metadata = f"""Image Index: {idx}
Attack Type: {attack_type}
Attack Goal: {attack_goal}
Source Model Label: {source_label}
Original Prediction: {original_prediction}
Adversarial Prediction: {adversarial_prediction}
Success: True
"""
        
        return metadata
