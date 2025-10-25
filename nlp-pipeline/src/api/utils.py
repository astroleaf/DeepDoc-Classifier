import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import re

logger = logging.getLogger(__name__)


def validate_text(text: str, min_length: int = 10, max_length: int = 5000) -> Tuple[bool, Optional[str]]:
    """
    Validate input text with detailed error messages.
    
    Args:
        text: Input text to validate
        min_length: Minimum acceptable text length
        max_length: Maximum acceptable text length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(text, str):
        return False, "Text must be a string"
    
    if not text or not text.strip():
        return False, "Text cannot be empty"
    
    text_length = len(text.strip())
    
    if text_length < min_length:
        return False, f"Text too short (minimum {min_length} characters)"
    
    if text_length > max_length:
        return False, f"Text too long (maximum {max_length} characters)"
    
    return True, None


def validate_batch_texts(texts: List[str], max_batch_size: int = 64) -> Tuple[bool, Optional[str]]:
    """
    Validate batch of texts.
    
    Args:
        texts: List of texts to validate
        max_batch_size: Maximum number of texts allowed in batch
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(texts, list):
        return False, "texts must be a list"
    
    if len(texts) == 0:
        return False, "texts list cannot be empty"
    
    if len(texts) > max_batch_size:
        return False, f"Batch size exceeds maximum ({max_batch_size})"
    
    for i, text in enumerate(texts):
        is_valid, error = validate_text(text)
        if not is_valid:
            return False, f"Text at index {i}: {error}"
    
    return True, None


def format_prediction_response(
    predictions: np.ndarray,
    class_names: List[str] = None,
    threshold: float = 0.5,
    include_all_probs: bool = False
) -> List[Dict]:
    """
    Format model predictions for API response.
    
    Args:
        predictions: Model prediction probabilities
        class_names: Optional list of class names
        threshold: Confidence threshold for warnings
        include_all_probs: Include all class probabilities
        
    Returns:
        List of formatted prediction dictionaries
    """
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    
    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    results = []
    for i, (pred_class, confidence) in enumerate(zip(predicted_classes, confidences)):
        result = {
            'class': int(pred_class),
            'confidence': float(confidence)
        }
        
        if class_names and pred_class < len(class_names):
            result['class_name'] = class_names[pred_class]
        
        if include_all_probs:
            result['all_probabilities'] = {
                class_names[j] if class_names else f'class_{j}': float(predictions[i][j])
                for j in range(len(predictions[i]))
            }
        
        if confidence < threshold:
            result['warning'] = 'Low confidence prediction'
            result['confidence_level'] = 'low'
        elif confidence < 0.75:
            result['confidence_level'] = 'medium'
        else:
            result['confidence_level'] = 'high'
        
        results.append(result)
    
    return results


def sanitize_text(text: str) -> str:
    """
    Sanitize input text to prevent injection attacks.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    # Remove any potential HTML/script tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Trim
    text = text.strip()
    
    return text


def create_error_response(error_type: str, message: str, details: Dict = None) -> Dict:
    """
    Create standardized error response.
    
    Args:
        error_type: Type of error (validation, processing, etc.)
        message: Error message
        details: Optional additional details
        
    Returns:
        Error response dictionary
    """
    response = {
        'error': True,
        'error_type': error_type,
        'message': message
    }
    
    if details:
        response['details'] = details
    
    return response


def calculate_confidence_stats(predictions: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics about prediction confidence.
    
    Args:
        predictions: Model predictions
        
    Returns:
        Dictionary of confidence statistics
    """
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    
    max_confidences = np.max(predictions, axis=1)
    
    stats = {
        'mean_confidence': float(np.mean(max_confidences)),
        'median_confidence': float(np.median(max_confidences)),
        'min_confidence': float(np.min(max_confidences)),
        'max_confidence': float(np.max(max_confidences)),
        'std_confidence': float(np.std(max_confidences))
    }
    
    return stats


def batch_texts(texts: List[str], batch_size: int) -> List[List[str]]:
    """
    Split texts into batches.
    
    Args:
        texts: List of texts
        batch_size: Size of each batch
        
    Returns:
        List of text batches
    """
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append(texts[i:i + batch_size])
    return batches


def log_prediction_metrics(
    text_length: int,
    latency_ms: float,
    confidence: float,
    predicted_class: int
) -> None:
    """
    Log detailed prediction metrics.
    
    Args:
        text_length: Length of input text
        latency_ms: Prediction latency in milliseconds
        confidence: Prediction confidence
        predicted_class: Predicted class
    """
    logger.debug(
        f"Prediction metrics - "
        f"text_length: {text_length}, "
        f"latency: {latency_ms:.2f}ms, "
        f"confidence: {confidence:.4f}, "
        f"class: {predicted_class}"
    )


def format_api_success_response(
    data: Dict,
    message: str = "Success",
    metadata: Dict = None
) -> Dict:
    """
    Create standardized success response.
    
    Args:
        data: Response data
        message: Success message
        metadata: Optional metadata
        
    Returns:
        Success response dictionary
    """
    response = {
        'success': True,
        'message': message,
        'data': data
    }
    
    if metadata:
        response['metadata'] = metadata
    
    return response
