"""
Feature extraction for ensemble models
"""
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
import pickle


class FeatureExtractor:
    """Extract various features from text for ensemble models"""
    
    def __init__(self, config: dict):
        self.config = config
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.get('tfidf_features', 5000),
            ngram_range=(1, 2),
            min_df=2
        )
        self.count_vectorizer = CountVectorizer(
            max_features=config.get('count_features', 5000),
            ngram_range=(1, 2)
        )
        self.word2vec_model = None
        self.fitted = False
    
    def fit(self, texts: List[str]):
        """Fit feature extractors on texts"""
        print("Fitting TF-IDF vectorizer...")
        self.tfidf_vectorizer.fit(texts)
        
        print("Fitting Count vectorizer...")
        self.count_vectorizer.fit(texts)
        
        # Train Word2Vec model
        print("Training Word2Vec model...")
        sentences = [text.split() for text in texts]
        self.word2vec_model = Word2Vec(
            sentences,
            vector_size=self.config.get('embedding_dim', 300),
            window=5,
            min_count=2,
            workers=4,
            epochs=10
        )
        
        self.fitted = True
        print("Feature extractors fitted successfully")
    
    def extract_tfidf(self, texts: List[str]) -> np.ndarray:
        """Extract TF-IDF features"""
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted first")
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def extract_count(self, texts: List[str]) -> np.ndarray:
        """Extract count features"""
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted first")
        return self.count_vectorizer.transform(texts).toarray()
    
    def extract_word2vec(self, texts: List[str]) -> np.ndarray:
        """Extract Word2Vec embeddings (average of word vectors)"""
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted first")
        
        embeddings = []
        dim = self.config.get('embedding_dim', 300)
        
        for text in texts:
            words = text.split()
            word_vectors = []
            
            for word in words:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            
            if word_vectors:
                # Average word vectors
                embedding = np.mean(word_vectors, axis=0)
            else:
                # Zero vector if no words found
                embedding = np.zeros(dim)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """Extract statistical text features"""
        features = []
        
        for text in texts:
            words = text.split()
            chars = list(text)
            
            feat = [
                len(words),  # Word count
                len(chars),  # Character count
                len(set(words)),  # Unique words
                np.mean([len(w) for w in words]) if words else 0,  # Avg word length
                sum(1 for c in chars if c.isupper()) / len(chars) if chars else 0,  # Uppercase ratio
                sum(1 for c in chars if c.isdigit()) / len(chars) if chars else 0,  # Digit ratio
                text.count('!'),  # Exclamation marks
                text.count('?'),  # Question marks
            ]
            features.append(feat)
        
        return np.array(features)
    
    def extract_all(self, texts: List[str]) -> Tuple[np.ndarray, ...]:
        """Extract all features"""
        tfidf = self.extract_tfidf(texts)
        count = self.extract_count(texts)
        w2v = self.extract_word2vec(texts)
        stats = self.extract_statistical_features(texts)
        
        return tfidf, count, w2v, stats
    
    def save(self, filepath: str):
        """Save feature extractor"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'count_vectorizer': self.count_vectorizer,
                'word2vec_model': self.word2vec_model,
                'config': self.config,
                'fitted': self.fitted
            }, f)
        print(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load feature extractor"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(data['config'])
        instance.tfidf_vectorizer = data['tfidf_vectorizer']
        instance.count_vectorizer = data['count_vectorizer']
        instance.word2vec_model = data['word2vec_model']
        instance.fitted = data['fitted']
        
        print(f"Feature extractor loaded from {filepath}")
        return instance