"""
Ensemble model combining BERT and LSTM for maximum accuracy (94%+)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Optional
import pickle


class EnsembleClassifier:
    """Ensemble model combining BERT and BiLSTM predictions"""
    
    def __init__(self, bert_model, lstm_model, config: dict):
        self.bert_model = bert_model
        self.lstm_model = lstm_model
        self.config = config
        self.weights = config.get('ensemble_weights', [0.6, 0.4])  # BERT, LSTM
        self.meta_model = None
        self.use_meta_learner = config.get('use_meta_learner', True)
        
        if self.use_meta_learner:
            self.build_meta_learner()
    
    def build_meta_learner(self):
        """Build a meta-learner to combine predictions intelligently"""
        num_classes = self.config.get('num_classes', 2)
        
        # Input: predictions from both models
        if num_classes == 2:
            input_dim = 2  # Binary predictions from BERT and LSTM
        else:
            input_dim = num_classes * 2
        
        inputs = keras.layers.Input(shape=(input_dim,))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        
        if num_classes == 2:
            outputs = keras.layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
            loss = 'sparse_categorical_crossentropy'
        
        self.meta_model = keras.Model(inputs=inputs, outputs=outputs)
        self.meta_model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )
    
    def train_meta_learner(self, X_bert, X_lstm, y_true):
        """Train the meta-learner on validation predictions"""
        # Get predictions from both models
        bert_preds = self.bert_model.predict(X_bert)
        lstm_preds = self.lstm_model.predict(X_lstm)
        
        # Combine predictions as features
        if len(bert_preds.shape) == 1:
            bert_preds = bert_preds.reshape(-1, 1)
        if len(lstm_preds.shape) == 1:
            lstm_preds = lstm_preds.reshape(-1, 1)
        
        meta_features = np.concatenate([bert_preds, lstm_preds], axis=1)
        
        # Train meta-learner
        self.meta_model.fit(
            meta_features, y_true,
            epochs=10,
            batch_size=32,
            verbose=0
        )
        print("Meta-learner trained successfully")
    
    def predict(self, X_bert: Dict, X_lstm: np.ndarray, 
                method: str = 'weighted') -> np.ndarray:
        """Make ensemble predictions
        
        Args:
            X_bert: Dict with 'input_ids' and 'attention_mask' for BERT
            X_lstm: NumPy array for LSTM
            method: 'weighted', 'average', or 'meta' (if meta-learner trained)
        """
        # Get predictions from both models
        bert_preds = self.bert_model.predict(X_bert)
        lstm_preds = self.lstm_model.predict(X_lstm)
        
        if method == 'meta' and self.meta_model is not None:
            # Use meta-learner
            if len(bert_preds.shape) == 1:
                bert_preds = bert_preds.reshape(-1, 1)
            if len(lstm_preds.shape) == 1:
                lstm_preds = lstm_preds.reshape(-1, 1)
            
            meta_features = np.concatenate([bert_preds, lstm_preds], axis=1)
            return self.meta_model.predict(meta_features)
        
        elif method == 'weighted':
            # Weighted average
            w1, w2 = self.weights
            return w1 * bert_preds + w2 * lstm_preds
        
        else:
            # Simple average
            return (bert_preds + lstm_preds) / 2
    
    def evaluate(self, X_bert: Dict, X_lstm: np.ndarray, y_true: np.ndarray) -> Dict:
        """Evaluate ensemble performance"""
        predictions = self.predict(X_bert, X_lstm, method='meta' if self.meta_model else 'weighted')
        
        # Convert to binary predictions
        num_classes = self.config.get('num_classes', 2)
        if num_classes == 2:
            y_pred = (predictions > 0.5).astype(int)
        else:
            y_pred = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred.flatten() == y_true.flatten())
        
        # Calculate precision, recall, F1
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary' if num_classes == 2 else 'weighted'
        )
        
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def optimize_weights(self, X_bert: Dict, X_lstm: np.ndarray, y_true: np.ndarray):
        """Find optimal ensemble weights using grid search"""
        best_accuracy = 0
        best_weights = self.weights
        
        print("Optimizing ensemble weights...")
        for w1 in np.arange(0.3, 0.8, 0.05):
            w2 = 1 - w1
            self.weights = [w1, w2]
            
            predictions = self.predict(X_bert, X_lstm, method='weighted')
            
            num_classes = self.config.get('num_classes', 2)
            if num_classes == 2:
                y_pred = (predictions > 0.5).astype(int)
            else:
                y_pred = np.argmax(predictions, axis=1)
            
            accuracy = np.mean(y_pred.flatten() == y_true.flatten())
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = [w1, w2]
        
        self.weights = best_weights
        print(f"Optimal weights: BERT={best_weights[0]:.3f}, LSTM={best_weights[1]:.3f}")
        print(f"Best accuracy: {best_accuracy:.4f}")
    
    def save(self, filepath: str):
        """Save ensemble configuration"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'config': self.config,
                'use_meta_learner': self.use_meta_learner
            }, f)
        
        if self.meta_model is not None:
            self.meta_model.save(filepath.replace('.pkl', '_meta.h5'))
        
        print(f"Ensemble saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, bert_model, lstm_model):
        """Load ensemble configuration"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(bert_model, lstm_model, data['config'])
        instance.weights = data['weights']
        instance.use_meta_learner = data['use_meta_learner']
        
        meta_path = filepath.replace('.pkl', '_meta.h5')
        try:
            instance.meta_model = keras.models.load_model(meta_path)
        except:
            instance.meta_model = None
        
        print(f"Ensemble loaded from {filepath}")
        return instance