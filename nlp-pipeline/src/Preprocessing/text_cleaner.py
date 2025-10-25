"""
Advanced text cleaning with parallel processing for 30% optimization
"""
import re
import unicodedata
from multiprocessing import Pool, cpu_count
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


class TextCleaner:
    """High-performance text cleaning with parallel processing"""
    
    def __init__(self, config: dict):
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.max_workers = config.get('max_workers', cpu_count())
        
        # Compiled regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def clean_single(self, text: str) -> str:
        """Clean a single text document"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs, emails, mentions, hashtags
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.mention_pattern.sub(' ', text)
        text = self.hashtag_pattern.sub(' ', text)
        
        # Lowercase if configured
        if self.config.get('lowercase', True):
            text = text.lower()
        
        # Remove numbers if configured
        if self.config.get('remove_numbers', False):
            text = self.number_pattern.sub(' ', text)
        
        # Remove punctuation if configured
        if self.config.get('remove_punctuation', False):
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords if configured
        if self.config.get('remove_stopwords', False):
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatization if configured
        if self.config.get('lemmatization', True):
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Filter by length
        min_len = self.config.get('min_word_length', 2)
        tokens = [t for t in tokens if len(t) >= min_len]
        
        return ' '.join(tokens)
    
    def clean_batch(self, texts: List[str], use_parallel: bool = True) -> List[str]:
        """Clean a batch of texts with optional parallel processing"""
        if not use_parallel or len(texts) < 1000:
            return [self.clean_single(t) for t in texts]
        
        # Parallel processing for large batches
        chunk_size = max(1, len(texts) // self.max_workers)
        with Pool(self.max_workers) as pool:
            cleaned = pool.map(self.clean_single, texts, chunksize=chunk_size)
        
        return cleaned
    
    def clean_and_validate(self, texts: List[str]) -> List[str]:
        """Clean texts and filter by length constraints"""
        cleaned = self.clean_batch(texts)
        
        min_len = self.config.get('min_document_length', 10)
        max_len = self.config.get('max_document_length', 5000)
        
        valid_texts = []
        for text in cleaned:
            word_count = len(text.split())
            if min_len <= word_count <= max_len:
                valid_texts.append(text)
        
        return valid_texts


class AdvancedTextCleaner(TextCleaner):
    """Extended cleaner with SpaCy for advanced NLP tasks"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        except OSError:
            print("SpaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_with_pos(self, text: str, keep_pos: Optional[List[str]] = None) -> str:
        """Clean text keeping only specified POS tags"""
        if not self.nlp:
            return self.clean_single(text)
        
        if keep_pos is None:
            keep_pos = ['NOUN', 'VERB', 'ADJ', 'ADV']
        
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.pos_ in keep_pos and not token.is_stop]
        
        return ' '.join(tokens)