import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_model import BidirectionalLSTMModel
from src.models.ensemble import EnsembleModel


class TestBidirectionalLSTMModel:
    """Test suite for BiLSTM model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.vocab_size = 1000
        self.embedding_dim = 64
        self.lstm_units = 128
        self.max_seq_length = 50
        self.num_classes = 3
        
        self.model_builder = BidirectionalLSTMModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            max_sequence_length=self.max_seq_length,
            num_classes=self.num_classes,
            dropout=0.3
        )
    
    def test_model_building(self):
        """Test model architecture building."""
        model = self.model_builder.build_model()
        
        assert model is not None
        assert isinstance(model, keras.Model)
    
    def test_model_input_shape(self):
        """Test model input shape."""
        model = self.model_builder.build_model()
        
        expected_input_shape = (None, self.max_seq_length)
        assert model.input_shape == expected_input_shape
    
    def test_model_output_shape(self):
        """Test model output shape."""
        model = self.model_builder.build_model()
        
        expected_output_shape = (None, self.num_classes)
        assert model.output_shape == expected_output_shape
    
    def test_model_compilation(self):
        """Test model compilation."""
        model = self.model_builder.build_model()
        self.model_builder.compile_model(learning_rate=1e-3)
        
        assert model.optimizer is not None
        assert model.loss is not None
    
    def test_model_prediction(self):
        """Test model prediction."""
        model = self.model_builder.build_model()
        self.model_builder.compile_model()
        
        # Create dummy input
        X = np.random.randint(0, self.vocab_size, (10, self.max_seq_length))
        predictions = model.predict(X, verbose=0)
        
        assert predictions.shape == (10, self.num_classes)
        # Predictions should sum to 1 (softmax)
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)
    
    def test_model_training(self):
        """Test model training."""
        model = self.model_builder.build_model()
        self.model_builder.compile_model()
        
        # Create dummy data
        X = np.random.randint(0, self.vocab_size, (100, self.max_seq_length))
        y = np.random.randint(0, self.num_classes, 100)
        
        # Train for 1 epoch
        history = model.fit(X, y, epochs=1, batch_size=32, verbose=0)
        
        assert 'loss' in history.history
        assert 'accuracy' in history.history


class TestEnsembleModel:
    """Test suite for Ensemble model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create mock models
        class MockModel:
            def predict(self, X):
                # Return random probabilities
                return np.random.random((len(X), 3))
        
        self.models = [MockModel() for _ in range(3)]
        self.weights = [0.5, 0.3, 0.2]
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        ensemble = EnsembleModel(self.models, self.weights)
        
        assert ensemble is not None
        assert len(ensemble.models) == 3
        assert len(ensemble.weights) == 3
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        ensemble = EnsembleModel(self.models, self.weights)
        
        X = np.random.random((10, 50))
        predictions = ensemble.predict(X)
        
        assert predictions is not None
        assert predictions.shape[0] == 10
    
    def test_weight_validation(self):
        """Test weight validation."""
        # Weights should sum to 1
        invalid_weights = [0.5, 0.3, 0.3]  # Sum = 1.1
        
        with pytest.raises(AssertionError):
            EnsembleModel(self.models, invalid_weights)
    
    def test_model_weight_mismatch(self):
        """Test model-weight count mismatch."""
        invalid_weights = [0.5, 0.5]  # Only 2 weights for 3 models
        
        with pytest.raises(AssertionError):
            EnsembleModel(self.models, invalid_weights)