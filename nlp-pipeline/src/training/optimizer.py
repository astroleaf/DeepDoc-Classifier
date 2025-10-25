import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any, Optional
import logging
from tqdm import tqdm
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class KerasTrainer:
    """Trainer for Keras/TensorFlow models with advanced optimizations."""
    
    def __init__(
        self,
        model: keras.Model,
        batch_size: int = 32,
        epochs: int = 10,
        validation_split: float = 0.1,
        early_stopping_patience: int = 3,
        use_mixed_precision: bool = True,
        save_best_only: bool = True,
        verbose: int = 1
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Keras model to train
            batch_size: Training batch size
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            early_stopping_patience: Patience for early stopping
            use_mixed_precision: Enable mixed precision training
            save_best_only: Save only the best model
            verbose: Verbosity level
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        # Enable mixed precision for 30% speedup
        if use_mixed_precision:
            try:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                logger.info("✓ Mixed precision training enabled (expected 30% speedup)")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        class_weights: Dict[int, float] = None
    ) -> Dict[str, Any]:
        """
        Train the model with optimizations.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            class_weights: Class weights for imbalanced datasets
            
        Returns:
            Training history dictionary
        """
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            logger.info(f"Using provided validation set: {len(X_val)} samples")
        else:
            logger.info(f"Using validation split: {self.validation_split}")
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING STARTED")
        logger.info("="*70)
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Steps per epoch: {len(X_train) // self.batch_size}")
        
        if class_weights:
            logger.info(f"Using class weights: {class_weights}")
        
        # Start training timer
        start_time = time.time()
        
        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            validation_split=self.validation_split if validation_data is None else 0,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=self.verbose
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        training_time_minutes = training_time / 60
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETED!")
        logger.info("="*70)
        logger.info(f"Total training time: {training_time_minutes:.2f} minutes")
        logger.info(f"Average time per epoch: {training_time_minutes/len(history.history['loss']):.2f} minutes")
        
        # Log best metrics
        self._log_best_metrics(history.history)
        
        return history.history
    
    def _setup_callbacks(self) -> list:
        """Setup training callbacks."""
        callbacks = []
        
        # Create models directory
        Path('data/models').mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        callbacks.append(early_stopping)
        logger.info(f"✓ Early stopping (patience={self.early_stopping_patience})")
        
        # Learning rate reduction
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(1, self.early_stopping_patience // 2),
            min_lr=1e-7,
            verbose=1,
            mode='min'
        )
        callbacks.append(reduce_lr)
        logger.info("✓ Learning rate scheduler")
        
        # Model checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(
            'data/models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=self.save_best_only,
            verbose=1,
            mode='max'
        )
        callbacks.append(checkpoint)
        logger.info("✓ Model checkpoint")
        
        # TensorBoard
        try:
            Path('logs/tensorboard').mkdir(parents=True, exist_ok=True)
            tensorboard = keras.callbacks.TensorBoard(
                log_dir='logs/tensorboard',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
            callbacks.append(tensorboard)
            logger.info("✓ TensorBoard logging")
        except Exception as e:
            logger.warning(f"TensorBoard not available: {e}")
        
        # CSV Logger
        try:
            csv_logger = keras.callbacks.CSVLogger(
                'logs/training_history.csv',
                append=False
            )
            callbacks.append(csv_logger)
            logger.info("✓ CSV logging")
        except Exception as e:
            logger.warning(f"CSV logger not available: {e}")
        
        return callbacks
    
    def _log_best_metrics(self, history: Dict[str, list]) -> None:
        """Log best training metrics."""
        if 'val_accuracy' in history:
            best_val_acc = max(history['val_accuracy'])
            best_epoch = history['val_accuracy'].index(best_val_acc) + 1
            logger.info(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
        
        if 'val_loss' in history:
            best_val_loss = min(history['val_loss'])
            best_epoch = history['val_loss'].index(best_val_loss) + 1
            logger.info(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of test metrics
        """
        logger.info("\n" + "="*70)
        logger.info("EVALUATION ON TEST SET")
        logger.info("="*70)
        logger.info(f"Test samples: {len(X_test)}")
        
        results = self.model.evaluate(X_test, y_test, verbose=self.verbose)
        
        metrics = dict(zip(self.model.metrics_names, results))
        
        logger.info("\nTest Results:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        return metrics


class PyTorchTrainer:
    """Trainer for PyTorch models with advanced optimizations."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        batch_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 2e-5,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 0
    ):
        """
        Initialize PyTorch trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on (cuda/cpu)
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Mixed precision scaler for 30% speedup
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Learning rate scheduler
        self.scheduler = None
        
        logger.info(f"PyTorch Trainer initialized on {device}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        if self.scaler:
            logger.info("✓ Mixed precision training enabled")
    
    def train(
        self,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset = None,
        save_path: str = 'data/models/pytorch_model.pt'
    ) -> Dict[str, list]:
        """
        Train the PyTorch model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            save_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup learning rate scheduler
        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        logger.info("\n" + "="*70)
        logger.info("PYTORCH TRAINING STARTED")
        logger.info("="*70)
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Total optimization steps: {total_steps}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            logger.info('='*70)
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_dataset is not None:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=4
                )
                val_loss, val_acc = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                logger.info(
                    f"\nEpoch {epoch + 1} Results:\n"
                    f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n"
                    f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(save_path)
                    logger.info(f"✓ Best model saved (accuracy: {val_acc:.4f})")
            else:
                logger.info(f"\nEpoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        training_time = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info("PYTORCH TRAINING COMPLETED!")
        logger.info("="*70)
        logger.info(f"Total training time: {training_time/60:.2f} minutes")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for i, batch in enumerate(progress_bar):
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)
            
            # Mixed precision forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient accumulation
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Update weights
            if (i + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
            
            # Calculate metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100 * correct / total
            progress_bar.set_postfix({
                'loss': f'{total_loss / (i + 1):.4f}',
                'acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in progress_bar:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': f'{total_loss / (len(progress_bar)):.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        val_loss = total_loss / len(val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def save_model(self, filepath: str) -> None:
        """Save model checkpoint."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Model loaded from {filepath}")