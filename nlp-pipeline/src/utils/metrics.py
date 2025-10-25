import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted',
    labels: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class ('micro', 'macro', 'weighted')
        labels: List of label indices to include
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0, labels=labels),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0, labels=labels),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0, labels=labels)
    }
    
    # Log metrics
    logger.info("\n" + "="*70)
    logger.info("CLASSIFICATION METRICS")
    logger.info("="*70)
    
    for key, value in metrics.items():
        logger.info(f"{key.upper():15s}: {value:.4f} ({value*100:.2f}%)")
    
    logger.info("\n" + "-"*70)
    logger.info("CONFUSION MATRIX")
    logger.info("-"*70)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    logger.info("\n" + str(cm))
    
    logger.info("\n" + "-"*70)
    logger.info("DETAILED CLASSIFICATION REPORT")
    logger.info("-"*70)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=