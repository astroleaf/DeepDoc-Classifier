"""
Main training script for the NLP pipeline
"""
import argparse
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys

# Add src to path
sys.path.append('src')

from preprocessing.text_cleaner import TextCleaner
from preprocessing.tokenizer import CustomTokenizer, BERTTokenizerWrapper
from models.lstm_model import BiLSTMClassifier, AttentionBiLSTM
from models.bert_model import BERTClassifier
from models.ensemble import EnsembleClassifier
from training.trainer import OptimizedTrainer


def load_data(filepath: str):
    """Load dataset from CSV"""
    print(f"Loading data from {filepath}...")
    
    # Support multiple formats
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        df = pd.read_json(filepath)
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Use CSV, JSON, or XLSX")
    
    print(f"Loaded {len(df)} samples")
    return df


def preprocess_data(df: pd.DataFrame, config: dict):
    """Preprocess text data"""
    print("\n" + "="*50)
    print("Preprocessing Data")
    print("="*50)
    
    # Initialize text cleaner
    cleaner = TextCleaner(config['preprocessing'])
    
    # Get text and label columns
    text_col = config['data'].get('text_column', 'text')
    label_col = config['data'].get('label_column', 'label')
    
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Columns {text_col} and {label_col} must exist in dataset")
    
    # Clean texts
    print("Cleaning texts...")
    texts = df[text_col].astype(str).tolist()
    cleaned_texts = cleaner.clean_and_validate(texts)
    
    # Get corresponding labels
    valid_indices = [i for i, text in enumerate(texts) 
                    if cleaner.clean_single(text) in cleaned_texts]
    labels = df[label_col].iloc[valid_indices].values
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    print(f"Valid samples: {len(cleaned_texts)}")
    print(f"Classes: {le.classes_}")
    
    return cleaned_texts, labels_encoded, le


def prepare_lstm_data(texts, labels, config):
    """Prepare data for LSTM model"""
    print("\nPreparing LSTM data...")
    
    # Initialize tokenizer
    tokenizer = CustomTokenizer(
        vocab_size=config['model'].get('vocab_size', 50000),
        max_length=config['model']['max_sequence_length']
    )
    
    # Fit tokenizer
    tokenizer.fit_on_texts(texts)
    
    # Encode texts
    X = tokenizer.encode(texts)
    
    # Save tokenizer
    os.makedirs('data/models', exist_ok=True)
    tokenizer.save('data/models/lstm_tokenizer.pkl')
    
    return X, tokenizer


def prepare_bert_data(texts, config):
    """Prepare data for BERT model"""
    print("\nPreparing BERT data...")
    
    # Initialize tokenizer
    tokenizer = BERTTokenizerWrapper(
        max_length=config['model']['max_sequence_length']
    )
    
    # Encode texts
    encodings = tokenizer.encode(texts, return_tensors=True)
    
    return encodings, tokenizer


def train_models(config: dict, X_lstm, X_bert, y, lstm_tokenizer):
    """Train LSTM, BERT, and ensemble models"""
    
    # Split data
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=1 - config['data']['train_split'],
        random_state=42,
        stratify=y
    )
    
    val_size = config['data']['val_split'] / (config['data']['val_split'] + config['data']['test_split'])
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_size,
        random_state=42,
        stratify=y[temp_idx]
    )
    
    print(f"\nDataset split:")
    print(f"Train: {len(train_idx)}")
    print(f"Val: {len(val_idx)}")
    print(f"Test: {len(test_idx)}")
    
    # Prepare splits
    X_lstm_train, X_lstm_val, X_lstm_test = X_lstm[train_idx], X_lstm[val_idx], X_lstm[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    # BERT splits
    X_bert_train = {k: v[train_idx] for k, v in X_bert.items()}
    X_bert_val = {k: v[val_idx] for k, v in X_bert.items()}
    X_bert_test = {k: v[test_idx] for k, v in X_bert.items()}
    
    # Update config with actual vocab size
    config['model']['vocab_size'] = lstm_tokenizer.get_vocab_size()
    config['model']['num_classes'] = len(np.unique(y))
    
    # Train LSTM Model
    print("\n" + "="*50)
    print("Training LSTM Model")
    print("="*50)
    
    lstm_config = config['model'].copy()
    lstm_model = AttentionBiLSTM(lstm_config)  # Use attention variant
    lstm_model.summary()
    
    lstm_trainer = OptimizedTrainer(lstm_model, config['training'])
    lstm_history = lstm_trainer.train(
        (X_lstm_train, y_train),
        (X_lstm_val, y_val)
    )
    
    # Save LSTM model
    lstm_model.save('data/models/lstm_model.h5')
    
    # Evaluate LSTM
    lstm_results = lstm_model.evaluate(X_lstm_test, y_test)
    print(f"\nLSTM Test Results: {lstm_results}")
    
    # Train BERT Model
    print("\n" + "="*50)
    print("Training BERT Model")
    print("="*50)
    
    bert_model = BERTClassifier(config['model'])
    bert_model.summary()
    
    bert_trainer = OptimizedTrainer(bert_model, config['training'])
    bert_history = bert_trainer.train(
        (X_bert_train, y_train),
        (X_bert_val, y_val)
    )
    
    # Fine-tune BERT (unfreeze last layers)
    print("\nFine-tuning BERT...")
    bert_model.unfreeze_bert(num_layers=4)
    bert_history_ft = bert_trainer.train(
        (X_bert_train, y_train),
        (X_bert_val, y_val)
    )
    
    # Save BERT model
    bert_model.save('data/models/bert_model.h5')
    
    # Evaluate BERT
    bert_results = bert_model.evaluate(X_bert_test, y_test)
    print(f"\nBERT Test Results: {bert_results}")
    
    # Create Ensemble
    print("\n" + "="*50)
    print("Creating Ensemble Model")
    print("="*50)
    
    ensemble = EnsembleClassifier(bert_model, lstm_model, config['model'])
    
    # Train meta-learner on validation set
    if config['model'].get('use_meta_learner', True):
        print("Training meta-learner...")
        ensemble.train_meta_learner(X_bert_val, X_lstm_val, y_val)
    
    # Optimize weights
    ensemble.optimize_weights(X_bert_val, X_lstm_val, y_val)
    
    # Evaluate ensemble
    print("\nEvaluating ensemble on test set...")
    ensemble_results = ensemble.evaluate(X_bert_test, X_lstm_test, y_test)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Ensemble Test Accuracy: {ensemble_results['accuracy']:.4f}")
    print(f"Precision: {ensemble_results['precision']:.4f}")
    print(f"Recall: {ensemble_results['recall']:.4f}")
    print(f"F1 Score: {ensemble_results['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(ensemble_results['confusion_matrix'])
    
    # Save ensemble
    ensemble.save('data/models/ensemble.pkl')
    
    return ensemble, ensemble_results


def main():
    parser = argparse.ArgumentParser(description='Train NLP models')
    parser.add_argument('--data-path', type=str, required=True, 
                       help='Path to training data (CSV/JSON/XLSX)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and preprocess data
    df = load_data(args.data_path)
    texts, labels, label_encoder = preprocess_data(df, config)
    
    # Prepare data for both models
    X_lstm, lstm_tokenizer = prepare_lstm_data(texts, labels, config)
    X_bert, bert_tokenizer = prepare_bert_data(texts, config)
    
    # Train models
    ensemble, results = train_models(config, X_lstm, X_bert, labels, lstm_tokenizer)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
    print("\nModels saved to:")
    print("- data/models/lstm_model.h5")
    print("- data/models/bert_model.h5")
    print("- data/models/ensemble.pkl")
    print("- data/models/lstm_tokenizer.pkl")


if __name__ == '__main__':
    main()