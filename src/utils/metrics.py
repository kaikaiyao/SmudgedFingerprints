"""Evaluation metrics for fingerprint robustness assessment."""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import Union, Tuple, Dict, Any


def calculate_tpr(y_true: Union[torch.Tensor, np.ndarray], 
                  y_pred: Union[torch.Tensor, np.ndarray],
                  positive_class: int = 1) -> float:
    """
    Calculate True Positive Rate (TPR) / Sensitivity / Recall.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        positive_class: Label of the positive class
        
    Returns:
        TPR value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
    fn = np.sum((y_true == positive_class) & (y_pred != positive_class))
    
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)


def calculate_tnr(y_true: Union[torch.Tensor, np.ndarray], 
                  y_pred: Union[torch.Tensor, np.ndarray],
                  positive_class: int = 1) -> float:
    """
    Calculate True Negative Rate (TNR) / Specificity.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        positive_class: Label of the positive class
        
    Returns:
        TNR value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    tn = np.sum((y_true != positive_class) & (y_pred != positive_class))
    fp = np.sum((y_true != positive_class) & (y_pred == positive_class))
    
    if tn + fp == 0:
        return 0.0
    
    return tn / (tn + fp)


def calculate_asr(y_true_clean: Union[torch.Tensor, np.ndarray],
                  y_pred_clean: Union[torch.Tensor, np.ndarray],
                  y_pred_attacked: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Attack Success Rate (ASR).
    ASR = (# correctly classified clean samples that become misclassified after attack) / (# correctly classified clean samples)
    
    Args:
        y_true_clean: Ground truth labels for clean samples
        y_pred_clean: Predictions on clean samples
        y_pred_attacked: Predictions on attacked samples
        
    Returns:
        ASR value
    """
    if isinstance(y_true_clean, torch.Tensor):
        y_true_clean = y_true_clean.cpu().numpy()
    if isinstance(y_pred_clean, torch.Tensor):
        y_pred_clean = y_pred_clean.cpu().numpy()
    if isinstance(y_pred_attacked, torch.Tensor):
        y_pred_attacked = y_pred_attacked.cpu().numpy()
    
    # Find correctly classified clean samples
    correct_clean = (y_true_clean == y_pred_clean)
    
    if np.sum(correct_clean) == 0:
        return 0.0
    
    # Among correctly classified clean samples, count how many become misclassified after attack
    successful_attacks = correct_clean & (y_true_clean != y_pred_attacked)
    
    return np.sum(successful_attacks) / np.sum(correct_clean)


def calculate_comprehensive_metrics(y_true: Union[torch.Tensor, np.ndarray],
                                   y_pred: Union[torch.Tensor, np.ndarray],
                                   y_scores: Union[torch.Tensor, np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_scores is not None and isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['tpr'] = calculate_tpr(y_true, y_pred)
    metrics['tnr'] = calculate_tnr(y_true, y_pred)
    
    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # AUC (if scores provided)
    if y_scores is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_scores)
            else:  # Multi-class
                metrics['auc'] = roc_auc_score(y_true, y_scores, multi_class='ovr')
        except ValueError:
            metrics['auc'] = 0.0
    
    return metrics


def calculate_robustness_metrics(clean_metrics: Dict[str, float],
                                attacked_metrics: Dict[str, float],
                                asr: float) -> Dict[str, float]:
    """
    Calculate robustness-specific metrics comparing clean and attacked performance.
    
    Args:
        clean_metrics: Metrics on clean samples
        attacked_metrics: Metrics on attacked samples
        asr: Attack Success Rate
        
    Returns:
        Dictionary of robustness metrics
    """
    robustness_metrics = {
        'asr': asr,
        'accuracy_drop': clean_metrics['accuracy'] - attacked_metrics['accuracy'],
        'tpr_drop': clean_metrics['tpr'] - attacked_metrics['tpr'],
        'tnr_drop': clean_metrics['tnr'] - attacked_metrics['tnr'],
        'f1_drop': clean_metrics['f1'] - attacked_metrics['f1'],
    }
    
    # Relative drops (percentage)
    for metric in ['accuracy', 'tpr', 'tnr', 'f1']:
        if clean_metrics[metric] > 0:
            robustness_metrics[f'{metric}_relative_drop'] = (
                (clean_metrics[metric] - attacked_metrics[metric]) / clean_metrics[metric]
            )
        else:
            robustness_metrics[f'{metric}_relative_drop'] = 0.0
    
    return robustness_metrics


def calculate_perceptual_distance(original: torch.Tensor, 
                                 perturbed: torch.Tensor,
                                 metric: str = "l2") -> float:
    """
    Calculate perceptual distance between original and perturbed images.
    
    Args:
        original: Original images tensor
        perturbed: Perturbed images tensor
        metric: Distance metric ("l2", "linf", "l1")
        
    Returns:
        Average distance
    """
    if metric == "l2":
        distance = torch.norm(original - perturbed, p=2, dim=[1, 2, 3])
    elif metric == "linf":
        distance = torch.norm(original - perturbed, p=float('inf'), dim=[1, 2, 3])
    elif metric == "l1":
        distance = torch.norm(original - perturbed, p=1, dim=[1, 2, 3])
    else:
        raise ValueError(f"Unknown distance metric: {metric}")
    
    return distance.mean().item()