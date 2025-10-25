"""
Flask API for NLP model inference with <150ms latency and 99.9% uptime
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import redis
import json
import time
import hashlib
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict, List
import numpy as np
import tensorflow as tf
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.text_cleaner import TextCleaner
from preprocessing.tokenizer import CustomTokenizer, BERTTokenizerWrapper
from models.ensemble import EnsembleClassifier
from models.bert_model import BERTClassifier
from models.lstm_model import BiLSTMClassifier
import yaml

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency', ['endpoint'])
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')

# Initialize Redis for caching
try:
    redis_client = redis.Redis(
        host=config['redis']['host'],
        port=config['redis']['port'],
        db=config['redis']['db'],
        decode_responses=True
    )
    redis_client.ping()
    print("Redis connected successfully")
except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_client = None

# Load models (singleton pattern for efficiency)
class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def initialize(self):
        """Initialize models and preprocessors"""
        if self.initialized:
            return
        
        print("Loading models...")
        
        # Load preprocessors
        self.text_cleaner = TextCleaner(config['preprocessing'])
        
        # Load tokenizers
        self.lstm_tokenizer = CustomTokenizer.load('data/models/lstm_tokenizer.pkl')
        self.bert_tokenizer = BERTTokenizerWrapper(max_length=config['model']['max_sequence_length'])
        
        # Load models
        self.bert_model = BERTClassifier.load('data/models/bert_model.h5', config['model'])
        self.lstm_model = BiLSTMClassifier.load('data/models/lstm_model.h5', config['model'])
        
        # Load ensemble
        self.ensemble = EnsembleClassifier.load(
            'data/models/ensemble.pkl',
            self.bert_model,
            self.lstm_model
        )
        
        self.initialized = True
        print("Models loaded successfully")
    
    def predict_single(self, text: str) -> Dict:
        """Predict single text with caching"""
        # Check cache
        if redis_client:
            cache_key = self._generate_cache_key(text)
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Preprocess
        cleaned_text = self.text_cleaner.clean_single(text)
        
        # Prepare inputs for both models
        bert_inputs = self.bert_tokenizer.encode([cleaned_text], return_tensors=True)
        lstm_inputs = self.lstm_tokenizer.encode([cleaned_text])
        
        # Get ensemble prediction
        prediction = self.ensemble.predict(bert_inputs, lstm_inputs, method='meta')
        
        # Format result
        num_classes = config['model']['num_classes']
        if num_classes == 2:
            result = {
                'prediction': int(prediction[0] > 0.5),
                'confidence': float(prediction[0]),
                'probabilities': {
                    'negative': float(1 - prediction[0]),
                    'positive': float(prediction[0])
                }
            }
        else:
            pred_class = int(np.argmax(prediction[0]))
            result = {
                'prediction': pred_class,
                'confidence': float(prediction[0][pred_class]),
                'probabilities': {i: float(p) for i, p in enumerate(prediction[0])}
            }
        
        # Cache result
        if redis_client:
            redis_client.setex(
                cache_key,
                config['api']['cache_ttl'],
                json.dumps(result)
            )
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict batch of texts efficiently"""
        # Preprocess all texts
        cleaned_texts = self.text_cleaner.clean_batch(texts)
        
        # Prepare inputs
        bert_inputs = self.bert_tokenizer.encode(cleaned_texts, return_tensors=True)
        lstm_inputs = self.lstm_tokenizer.encode(cleaned_texts)
        
        # Get predictions
        predictions = self.ensemble.predict(bert_inputs, lstm_inputs, method='meta')
        
        # Format results
        results = []
        num_classes = config['model']['num_classes']
        
        for i, pred in enumerate(predictions):
            if num_classes == 2:
                result = {
                    'text': texts[i],
                    'prediction': int(pred > 0.5),
                    'confidence': float(pred),
                    'probabilities': {
                        'negative': float(1 - pred),
                        'positive': float(pred)
                    }
                }
            else:
                pred_class = int(np.argmax(pred))
                result = {
                    'text': texts[i],
                    'prediction': pred_class,
                    'confidence': float(pred[pred_class]),
                    'probabilities': {i: float(p) for i, p in enumerate(pred)}
                }
            results.append(result)
        
        return results
    
    @staticmethod
    def _generate_cache_key(text: str) -> str:
        """Generate cache key from text"""
        return f"pred:{hashlib.md5(text.encode()).hexdigest()}"

# Initialize model manager
model_manager = ModelManager()

# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status=200).inc()
    
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'models_loaded': model_manager.initialized
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    start_time = time.time()
    
    try:
        # Validate request
        data = request.get_json()
        if not data or 'text' not in data:
            REQUEST_COUNT.labels(method='POST', endpoint='/predict', status=400).inc()
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        
        if not text or not isinstance(text, str):
            REQUEST_COUNT.labels(method='POST', endpoint='/predict', status=400).inc()
            return jsonify({'error': 'Invalid text'}), 400
        
        # Make prediction
        result = model_manager.predict_single(text)
        
        # Track metrics
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint='/predict').observe(latency)
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status=200).inc()
        PREDICTION_COUNT.inc()
        
        result['latency_ms'] = round(latency * 1000, 2)
        
        return jsonify(result), 200
    
    except Exception as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status=500).inc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    start_time = time.time()
    
    try:
        # Validate request
        data = request.get_json()
        if not data or 'texts' not in data:
            REQUEST_COUNT.labels(method='POST', endpoint='/predict/batch', status=400).inc()
            return jsonify({'error': 'Missing texts field'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or not texts:
            REQUEST_COUNT.labels(method='POST', endpoint='/predict/batch', status=400).inc()
            return jsonify({'error': 'texts must be a non-empty list'}), 400
        
        # Check batch size limit
        max_batch_size = config['api']['max_batch_size']
        if len(texts) > max_batch_size:
            REQUEST_COUNT.labels(method='POST', endpoint='/predict/batch', status=400).inc()
            return jsonify({'error': f'Batch size exceeds limit of {max_batch_size}'}), 400
        
        # Make predictions
        results = model_manager.predict_batch(texts)
        
        # Track metrics
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint='/predict/batch').observe(latency)
        REQUEST_COUNT.labels(method='POST', endpoint='/predict/batch', status=200).inc()
        PREDICTION_COUNT.inc(len(texts))
        
        return jsonify({
            'results': results,
            'count': len(results),
            'latency_ms': round(latency * 1000, 2),
            'avg_latency_per_text_ms': round((latency * 1000) / len(texts), 2)
        }), 200
    
    except Exception as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/predict/batch', status=500).inc()
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Get cache statistics"""
    if not redis_client:
        return jsonify({'error': 'Redis not available'}), 503
    
    try:
        info = redis_client.info('stats')
        return jsonify({
            'total_connections': info.get('total_connections_received', 0),
            'total_commands': info.get('total_commands_processed', 0),
            'keyspace_hits': info.get('keyspace_hits', 0),
            'keyspace_misses': info.get('keyspace_misses', 0),
            'hit_rate': round(
                info.get('keyspace_hits', 0) / 
                max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1) * 100, 
                2
            )
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear prediction cache"""
    if not redis_client:
        return jsonify({'error': 'Redis not available'}), 503
    
    try:
        # Clear only prediction keys
        keys = redis_client.keys('pred:*')
        if keys:
            redis_client.delete(*keys)
        
        return jsonify({
            'status': 'success',
            'keys_cleared': len(keys)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.before_first_request
def initialize():
    """Initialize models before first request"""
    model_manager.initialize()

@app.errorhandler(404)
def not_found(error):
    REQUEST_COUNT.labels(method=request.method, endpoint=request.path, status=404).inc()
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    REQUEST_COUNT.labels(method=request.method, endpoint=request.path, status=500).inc()
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize models
    model_manager.initialize()
    
    # Run app
    app.run(
        host=config['api']['host'],
        port=config['api']['port'],
        debug=config['api']['debug']
    )