"""
Embedding extraction for text and image data.
"""

import logging
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extract embeddings from text and image data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize embedding extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.embedding_config = config["embeddings"]
        
        self.text_model = None
        self.text_tokenizer = None
        self._initialize_text_model()
        
        logger.info("Embedding extractor initialized")
    
    def _initialize_text_model(self):
        """Initialize text embedding model."""
        try:
            model_name = self.embedding_config["model"]
            self.text_model = SentenceTransformer(model_name)
            
            device = self.embedding_config["device"]
            self.text_model.to(device)
            
            logger.info(f"Text embedding model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load text embedding model: {e}")
            self.text_model = None
    
    def extract_text_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Extract embeddings from text.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            Text embeddings
        """
        if self.text_model is None:
            logger.warning("Text model not available, returning dummy embeddings")
            if isinstance(texts, str):
                texts = [texts]
            return np.random.randn(len(texts), self.config["index"]["dimension"])
        
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = self.text_model.encode(
                texts,
                batch_size=self.embedding_config["batch_size"],
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to extract text embeddings: {e}")
        
            if isinstance(texts, str):
                texts = [texts]
            return np.random.randn(len(texts), self.config["index"]["dimension"])
    
    def extract_image_embeddings(self, image_features: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from image features.
        
        Args:
            image_features: Image feature vectors
            
        Returns:
            Image embeddings
        """
        try:
            target_dim = self.config["index"]["dimension"]
            
            if image_features.shape[-1] != target_dim:
                #Use PCA or simple projection for dimensionality reduction
                if image_features.shape[-1] > target_dim:
                    projection_matrix = np.random.randn(image_features.shape[-1], target_dim)
                    projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=0)
                    embeddings = np.dot(image_features, projection_matrix)
                else:
                    #Pad with zeros
                    embeddings = np.zeros((image_features.shape[0], target_dim))
                    embeddings[:, :image_features.shape[-1]] = image_features
            
            else:
                embeddings = image_features
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to extract image embeddings: {e}")
           
            return np.random.randn(image_features.shape[0], self.config["index"]["dimension"])
    
    def extract_combined_embeddings(self, text_data: Union[str, List[str]], 
                                   image_features: np.ndarray) -> np.ndarray:
        """
        Extract combined embeddings from text and image data.
        
        Args:
            text_data: Text data
            image_features: Image feature vectors
            
        Returns:
            Combined embeddings
        """
        try:
            text_embeddings = self.extract_text_embeddings(text_data)
            
            image_embeddings = self.extract_image_embeddings(image_features)
            
            #Combine embeddings (simple concatenation or weighted sum)
            combined_embeddings = (text_embeddings + image_embeddings) / 2
            
            return combined_embeddings
            
        except Exception as e:
            logger.error(f"Failed to extract combined embeddings: {e}")
            # Return dummy embeddings
            if isinstance(text_data, str):
                text_data = [text_data]
            return np.random.randn(len(text_data), self.config["index"]["dimension"])
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Normalized embeddings
        """
        try:
            # L2 normalization
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            normalized_embeddings = embeddings / norms
            
            return normalized_embeddings
            
        except Exception as e:
            logger.error(f"Failed to normalize embeddings: {e}")
            return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.config["index"]["dimension"]
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            Model information dictionary
        """
        info = {
            "text_model": self.embedding_config["model"],
            "embedding_dimension": self.get_embedding_dimension(),
            "max_length": self.embedding_config["max_length"],
            "batch_size": self.embedding_config["batch_size"],
            "device": self.embedding_config["device"]
        }
        
        if self.text_model is not None:
            info["text_model_loaded"] = True
        else:
            info["text_model_loaded"] = False
        
        return info 