"""
Custom tokenizer with vocabulary management for LSTM models
"""
import pickle
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer


class CustomTokenizer:
    """Custom tokenizer for LSTM models with vocabulary management"""
    
    def __init__(self, vocab_size: int = 50000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_counts = Counter()
        self.fitted = False
    
    def fit_on_texts(self, texts: List[str]):
        """Build vocabulary from texts"""
        # Count word frequencies
        for text in texts:
            words = text.split()
            self.word_counts.update(words)
        
        # Keep most common words
        most_common = self.word_counts.most_common(self.vocab_size - 4)
        
        # Build vocabulary
        for idx, (word, count) in enumerate(most_common, start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.fitted = True
        print(f"Vocabulary built: {len(self.word2idx)} unique tokens")
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert texts to sequences of integers"""
        if not self.fitted:
            raise ValueError("Tokenizer must be fitted before encoding")
        
        sequences = []
        for text in texts:
            words = text.split()
            seq = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>
            sequences.append(seq)
        
        return sequences
    
    def pad_sequences(self, sequences: List[List[int]], 
                     padding: str = 'post',
                     truncating: str = 'post') -> np.ndarray:
        """Pad sequences to uniform length"""
        return pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding=padding,
            truncating=truncating,
            value=0  # <PAD> token
        )
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to padded sequences"""
        sequences = self.texts_to_sequences(texts)
        return self.pad_sequences(sequences)
    
    def decode(self, sequence: List[int]) -> str:
        """Decode sequence back to text"""
        words = [self.idx2word.get(idx, '<UNK>') for idx in sequence if idx != 0]
        return ' '.join(words)
    
    def save(self, filepath: str):
        """Save tokenizer to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'vocab_size': self.vocab_size,
                'max_length': self.max_length,
                'fitted': self.fitted
            }, f)
        print(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'], max_length=data['max_length'])
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = data['idx2word']
        tokenizer.word_counts = data['word_counts']
        tokenizer.fitted = data['fitted']
        
        print(f"Tokenizer loaded from {filepath}")
        return tokenizer
    
    def get_vocab_size(self) -> int:
        """Get actual vocabulary size"""
        return len(self.word2idx)


class BERTTokenizerWrapper:
    """Wrapper for BERT tokenizer with consistent interface"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.fitted = True
    
    def fit_on_texts(self, texts: List[str]):
        """BERT tokenizer doesn't need fitting"""
        pass
    
    def encode(self, texts: List[str], return_tensors: bool = False) -> Dict:
        """Encode texts for BERT model"""
        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf' if return_tensors else None
        )
        return encoding
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def save(self, filepath: str):
        """Save tokenizer"""
        self.tokenizer.save_pretrained(filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer"""
        instance = cls.__new__(cls)
        instance.tokenizer = BertTokenizer.from_pretrained(filepath)
        instance.max_length = 512
        instance.fitted = True
        return instance
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.tokenizer.vocab_size