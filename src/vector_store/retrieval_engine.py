"""
Retrieval engine for searching and retrieving similar cases.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .faiss_index import FAISSIndex
from .embedding_extractor import EmbeddingExtractor

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Retrieval engine for searching and retrieving similar cases.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize retrieval engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.retrieval_config = config["retrieval"]
        self.search_config = config["search"]
        
        self.faiss_index = FAISSIndex(config)
        self.embedding_extractor = EmbeddingExtractor(config)
        
        logger.info("Retrieval engine initialized")
    
    def search_similar_cases(self, query_embeddings: np.ndarray, 
                           filters: Optional[Dict[str, Any]] = None,
                           k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar cases.
        
        Args:
            query_embeddings: Query embeddings
            filters: Metadata filters
            k: Number of results to retrieve
            
        Returns:
            List of similar cases with metadata
        """
        try:
            if k is None:
                k = self.retrieval_config["k"]
            
            if filters and self.search_config["hybrid_search"]:
                results = self.faiss_index.filter_search(query_embeddings, filters, k)
            else:
                results = self.faiss_index.search_with_metadata(query_embeddings, k)
            
            threshold = self.retrieval_config["similarity_threshold"]
            filtered_results = []
            
            for query_results in results:
                filtered_query_results = []
                for result in query_results:
                    if result["similarity"] >= threshold:
                        filtered_query_results.append(result)
                filtered_results.append(filtered_query_results)
            
            return filtered_results[0] if filtered_results else [] 
            
        except Exception as e:
            logger.error(f"Failed to search similar cases: {e}")
            return []
    
    def search_by_text(self, text: str, filters: Optional[Dict[str, Any]] = None,
                      k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search by text query.
        
        Args:
            text: Text query
            filters: Metadata filters
            k: Number of results to retrieve
            
        Returns:
            List of similar cases
        """
        try:
            text_embeddings = self.embedding_extractor.extract_text_embeddings(text)
            
            results = self.search_similar_cases(text_embeddings, filters, k)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search by text: {e}")
            return []
    
    def search_by_image_features(self, image_features: np.ndarray,
                                filters: Optional[Dict[str, Any]] = None,
                                k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search by image features.
        
        Args:
            image_features: Image feature vectors
            filters: Metadata filters
            k: Number of results to retrieve
            
        Returns:
            List of similar cases
        """
        try:
            image_embeddings = self.embedding_extractor.extract_image_embeddings(image_features)
            
            results = self.search_similar_cases(image_embeddings, filters, k)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search by image features: {e}")
            return []
    
    def search_by_combined_query(self, text: str, image_features: np.ndarray,
                                filters: Optional[Dict[str, Any]] = None,
                                k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search by combined text and image query.
        
        Args:
            text: Text query
            image_features: Image feature vectors
            filters: Metadata filters
            k: Number of results to retrieve
            
        Returns:
            List of similar cases
        """
        try:
            combined_embeddings = self.embedding_extractor.extract_combined_embeddings(
                text, image_features
            )
            
            results = self.search_similar_cases(combined_embeddings, filters, k)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search by combined query: {e}")
            return []
    
    def get_case_details(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific case.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Case details or None
        """
        try:
            return {
                "case_id": case_id,
                "patient_id": "unknown",
                "study_date": "unknown",
                "modality": "unknown",
                "body_region": "unknown",
                "findings": "No findings available",
                "impression": "No impression available"
            }
            
        except Exception as e:
            logger.error(f"Failed to get case details: {e}")
            return None
    
    def get_similarity_score(self, query_embeddings: np.ndarray, 
                           case_embeddings: np.ndarray) -> float:
        """
        Calculate similarity score between query and case embeddings.
        
        Args:
            query_embeddings: Query embeddings
            case_embeddings: Case embeddings
            
        Returns:
            Similarity score
        """
        try:
            query_norm = query_embeddings / np.linalg.norm(query_embeddings)
            case_norm = case_embeddings / np.linalg.norm(case_embeddings)
            
            similarity = np.dot(query_norm, case_norm)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity score: {e}")
            return 0.0
    
    def rank_results(self, results: List[Dict[str, Any]], 
                    ranking_criteria: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Rank search results based on multiple criteria.
        
        Args:
            results: Search results
            ranking_criteria: Ranking criteria and weights
            
        Returns:
            Ranked results
        """
        try:
            if not ranking_criteria:
                ranking_criteria = {"similarity": 1.0}
            
            for result in results:
                composite_score = 0.0
                
                for criterion, weight in ranking_criteria.items():
                    if criterion == "similarity":
                        composite_score += weight * result.get("similarity", 0.0)
                    elif criterion == "recency":
                      
                        pass
                    elif criterion == "relevance":
                        pass
                
                result["composite_score"] = composite_score
            
            ranked_results = sorted(results, key=lambda x: x["composite_score"], reverse=True)
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Failed to rank results: {e}")
            return results
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get search statistics.
        
        Returns:
            Search statistics dictionary
        """
        try:
            stats = {
                "total_cases": self.faiss_index.get_index_stats()["total_vectors"],
                "index_type": self.faiss_index.get_index_stats()["index_type"],
                "embedding_dimension": self.embedding_extractor.get_embedding_dimension(),
                "similarity_threshold": self.retrieval_config["similarity_threshold"],
                "max_distance": self.retrieval_config["max_distance"]
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get search statistics: {e}")
            return {}
    
    def add_case_to_index(self, case_embeddings: np.ndarray, case_metadata: Dict[str, Any]):
        """
        Add a new case to the index.
        
        Args:
            case_embeddings: Case embeddings
            case_metadata: Case metadata
        """
        try:
            self.faiss_index.add_embeddings(case_embeddings, [case_metadata])
            logger.info("Case added to index")
            
        except Exception as e:
            logger.error(f"Failed to add case to index: {e}")
    
    def remove_case_from_index(self, case_id: str):
        """
        Remove a case from the index.
        
        Args:
            case_id: Case identifier
        """
        try:
            metadata = self.faiss_index.metadata
            case_indices = [i for i, meta in enumerate(metadata) if meta.get("case_id") == case_id]
            
            if case_indices:
                self.faiss_index.remove_vectors(case_indices)
                logger.info(f"Case {case_id} removed from index")
            else:
                logger.warning(f"Case {case_id} not found in index")
                
        except Exception as e:
            logger.error(f"Failed to remove case from index: {e}")
    
    def update_case_in_index(self, case_id: str, new_embeddings: np.ndarray, 
                           new_metadata: Dict[str, Any]):
        """
        Update a case in the index.
        
        Args:
            case_id: Case identifier
            new_embeddings: New case embeddings
            new_metadata: New case metadata
        """
        try:
            self.remove_case_from_index(case_id)
            
            new_metadata["case_id"] = case_id
            self.add_case_to_index(new_embeddings, new_metadata)
            
            logger.info(f"Case {case_id} updated in index")
            
        except Exception as e:
            logger.error(f"Failed to update case in index: {e}") 
            