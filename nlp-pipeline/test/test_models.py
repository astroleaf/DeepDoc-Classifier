import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.tokenizer import NLPTokenizer
from src.preprocessing.feature_extractor import FeatureExtractor


class TestTextCleaner:
    """Test suite for TextCleaner class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.cleaner = TextCleaner(
            lowercase=True,
            remove_stopwords=True,
            lemmatize=True
        )
    
    def test_basic_cleaning(self):
        """Test basic text cleaning functionality."""
        text = "This is a TEST document!  With EXTRA   spaces."
        cleaned = self.cleaner.clean_text(text)
        
        assert cleaned is not None
        assert len(cleaned) > 0
        assert cleaned.islower()
        assert "  " not in cleaned  # No double spaces
    
    def test_url_removal(self):
        """Test URL removal."""
        text = "Check this link: https://example.com/page?param=value"
        cleaned = self.cleaner.clean_text(text)
        
        assert "https://" not in cleaned
        assert "example.com" not in cleaned
    
    def test_email_removal(self):
        """Test email removal."""
        text = "Contact us at support@example.com for help"
        cleaned = self.cleaner.clean_text(text)
        
        assert "@" not in cleaned
        assert "support@example.com" not in cleaned
    
    def test_stopword_removal(self):
        """Test stopword removal."""
        text = "The quick brown fox jumps over the lazy dog"
        cleaned = self.cleaner.clean_text(text)
        
        # Common stopwords should be removed
        assert "the" not in cleaned.lower()
        assert "over" not in cleaned.lower()
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        cleaned = self.cleaner.clean_text("")
        assert cleaned == ""
        
        cleaned = self.cleaner.clean_text("   ")
        assert len(cleaned) == 0
    
    def test_non_string_input(self):
        """Test handling of non-string input."""
        cleaned = self.cleaner.clean_text(None)
        assert cleaned == ""
        
        cleaned = self.cleaner.clean_text(123)
        assert cleaned == ""
    
    def test_batch_cleaning(self):
        """Test batch text cleaning."""
        texts = [
            "First document with URL https://test.com",
            "Second document WITH CAPS",
            "Third document with stopwords the and a"
        ]
        
        cleaned_texts = self.cleaner.clean_batch(texts, n_jobs=1)
        
        assert len(cleaned_texts) == len(texts)
        assert all(isinstance(text, str) for text in cleaned_texts)
        assert all(text.islower() for text in cleaned_texts if text)
    
    def test_unicode_normalization(self):
        """Test unicode normalization."""
        text = "café résumé naïve"
        cleaned = self.cleaner.clean_text(text)
        
        assert cleaned is not None
        assert len(cleaned) > 0
    
    def test_special_characters(self):
        """Test handling of special characters."""
        text = "Text with symbols: @#$%^&*()"
        cleaned = self.cleaner.clean_text(text)
        
        assert cleaned is not None


class TestNLPTokenizer:
    """Test suite for NLPTokenizer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tokenizer = NLPTokenizer(
            max_vocab_size=1000,
            max_sequence_length=50
        )
        
        # Sample training texts
        self.train_texts = [
            "this is the first document",
            "this document is the second document",
            "and this is the third one",
            "is this the first document"
        ]
    
    def test_tokenizer_fit(self):
        """Test tokenizer fitting."""
        self.tokenizer.fit(self.train_texts)
        
        vocab_size = self.tokenizer.get_vocab_size()
        assert vocab_size > 0
        assert vocab_size <= 1000  # Should respect max_vocab_size
    
    def test_texts_to_sequences(self):
        """Test text to sequence conversion."""
        self.tokenizer.fit(self.train_texts)
        sequences = self.tokenizer.texts_to_sequences(self.train_texts)
        
        assert sequences is not None
        assert sequences.shape[0] == len(self.train_texts)
        assert sequences.shape[1] == 50  # max_sequence_length
        assert sequences.dtype == np.int32 or sequences.dtype == np.int64
    
    def test_sequence_padding(self):
        """Test sequence padding."""
        self.tokenizer.fit(self.train_texts)
        sequences = self.tokenizer.texts_to_sequences(["short", "this is a longer text"])
        
        # All sequences should have same length
        assert sequences.shape[1] == 50
    
    def test_oov_token_handling(self):
        """Test out-of-vocabulary token handling."""
        self.tokenizer.fit(self.train_texts)
        
        # Text with unseen words
        new_text = ["completely unknown words here"]
        sequences = self.tokenizer.texts_to_sequences(new_text)
        
        assert sequences is not None
        assert sequences.shape[0] == 1
    
    def test_get_word_index(self):
        """Test word index retrieval."""
        self.tokenizer.fit(self.train_texts)
        word_index = self.tokenizer.get_word_index()
        
        assert isinstance(word_index, dict)
        assert len(word_index) > 0
        assert "document" in word_index
    
    def test_empty_text_handling(self):
        """Test handling of empty texts."""
        self.tokenizer.fit(self.train_texts)
        sequences = self.tokenizer.texts_to_sequences([""])
        
        assert sequences is not None
        assert sequences.shape[0] == 1


class TestFeatureExtractor:
    """Test suite for FeatureExtractor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = FeatureExtractor(
            max_features=100,
            ngram_range=(1, 2)
        )
        
        self.texts = [
            "machine learning is great",
            "deep learning neural networks",
            "natural language processing",
            "machine learning algorithms"
        ]
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        features = self.extractor.fit_transform(self.texts)
        
        assert features is not None
        assert features.shape[0] == len(self.texts)
        assert features.shape[1] <= 100  # max_features
        assert features.dtype == np.float64 or features.dtype == np.float32
    
    def test_transform(self):
        """Test transform method."""
        self.extractor.fit_transform(self.texts)
        
        new_texts = ["machine learning models"]
        features = self.extractor.transform(new_texts)
        
        assert features is not None
        assert features.shape[0] == len(new_texts)
    
    def test_feature_range(self):
        """Test that feature values are in valid range."""
        features = self.extractor.fit_transform(self.texts)
        
        # TF-IDF values should be between 0 and 1
        assert np.all(features >= 0)
        assert np.all(features <= 1)
    
    def test_ngram_extraction(self):
        """Test n-gram feature extraction."""
        features = self.extractor.fit_transform(self.texts)
        
        # Should extract both unigrams and bigrams
        assert features.shape[1] > len(set(" ".join(self.texts).split()))