"""
Target selection utilities for forgery attacks.
"""

import torch
import numpy as np
from typing import Optional


class TargetSelector:
    """Handles target selection for forgery attacks."""
    
    @staticmethod
    def select_random_targets(
        y_labels: torch.Tensor, 
        num_classes: int, 
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Select random target classes for each image, ensuring they're different from the original.
        
        Args:
            y_labels: Original class labels
            num_classes: Total number of classes available
            seed: Random seed for reproducibility
            
        Returns:
            Tensor of target class labels
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize targets as copy of original labels
        targets = y_labels.clone()
        
        # For each image, select a random target that's different from original
        for i in range(len(targets)):
            # Get all possible classes except the original
            possible_targets = list(range(num_classes))
            possible_targets.remove(y_labels[i].item())
            
            # Randomly select one
            if possible_targets:
                targets[i] = torch.randint(0, len(possible_targets), (1,)).item()
                targets[i] = possible_targets[targets[i].item()]
        
        return targets
    
    @staticmethod
    def validate_target_class(*args, **kwargs) -> bool:
        """Deprecated: fixed targets removed. Always returns False."""
        return False
