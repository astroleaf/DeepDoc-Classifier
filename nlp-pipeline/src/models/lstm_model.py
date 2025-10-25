"""
Bidirectional LSTM model for text classification
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Optional


class BiLSTMClassifier:
    """Bidirectional LSTM model for text classification"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.build_model()
    
    def build_model(self):
        """Build Bidirectional LSTM architecture"""
        vocab_size = self.config.get('vocab_size', 50000)
        embedding_dim = self.config.get('embedding_dim', 300)
        max_length = self.config.get('max_sequence_length', 512)
        lstm_units = self.config.get('lstm_units', 256)
        dropout = self.config.get('dropout', 0.3)
        num_classes = self.config.get('num_classes', 2)
        
        # Input layer
        inputs = layers.Input(shape=(max_length,), name='input')
        
        # Embedding layer with masking
        x = layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_length,
            mask_zero=True,
            name='embedding'
        )(inputs)
        
        # Spatial dropout for regularization
        x = layers.SpatialDropout1D(dropout)(x)
        
        # First Bidirectional LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout),
            name='bilstm_1'
        )(x)
        
        # Second Bidirectional LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(lstm_units // 2, return_sequences=False, dropout=dropout),
            name='bilstm_2'
        )(x)
        
        # Dense layers with batch normalization
        x = layers.Dense(lstm_units, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        x = layers.Dense(lstm_units // 2, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        # Output layer
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_Classifier')
        
        # Compile model
        self.compile_model()
    
    def compile_model(self):
        """Compile model with optimizer and loss"""
        num_classes = self.config.get('num_classes', 2)
        learning_rate = self.config.get('learning_rate', 1e-3)
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        if num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()
    
    def train(self, X_train, y_train, X_val, y_val, callbacks: Optional[list] = None):
        """Train the model"""
        batch_size = self.config.get('batch_size', 32)
        epochs = self.config.get('epochs', 10)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks or [],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        return self.model.evaluate(X_test, y_test)
    
    def save(self, filepath: str):
        """Save model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, config: dict):
        """Load model"""
        instance = cls(config)
        instance.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return instance


class AttentionBiLSTM(BiLSTMClassifier):
    """BiLSTM with attention mechanism for better performance"""
    
    def build_model(self):
        """Build BiLSTM with attention"""
        vocab_size = self.config.get('vocab_size', 50000)
        embedding_dim = self.config.get('embedding_dim', 300)
        max_length = self.config.get('max_sequence_length', 512)
        lstm_units = self.config.get('lstm_units', 256)
        dropout = self.config.get('dropout', 0.3)
        num_classes = self.config.get('num_classes', 2)
        
        # Input layer
        inputs = layers.Input(shape=(max_length,), name='input')
        
        # Embedding layer
        x = layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_length,
            mask_zero=True
        )(inputs)
        
        x = layers.SpatialDropout1D(dropout)(x)
        
        # Bidirectional LSTM with return sequences for attention
        lstm_out = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)
        )(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(lstm_out)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(lstm_units * 2)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention weights
        attended = layers.Multiply()([lstm_out, attention])
        attended = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)
        
        # Dense layers
        x = layers.Dense(lstm_units, activation='relu')(attended)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        # Output layer
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='AttentionBiLSTM')
        self.compile_model()