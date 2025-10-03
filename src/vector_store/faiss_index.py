"""
FAISS index implementation for vector storage and retrieval.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import faiss
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS index wrapper for medical image and text embeddings.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize FAISS index.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.index_config = config["index"]
        self.indexing_config = config["indexing"]
        
        self.index = None
        self.metadata = []
        self.dimension = self.index_config["dimension"]
        
        os.makedirs(os.path.dirname(self.indexing_config["index_path"]), exist_ok=True)
        
        self._load_index()
        
        logger.info(f"FAISS index initialized with dimension: {self.dimension}")
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        index_path = self.indexing_config["index_path"]
        metadata_path = self.indexing_config["metadata_path"]
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)
                
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                logger.info(f"Loaded existing index with {len(self.metadata)} entries")
                
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index."""
        index_type = self.index_config["type"]
        
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
        elif index_type == "IVFFlat":
            # Create quantizer
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = self.index_config["nlist"]
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
        elif index_type == "IVFPQ":
            # Create quantizer
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = self.index_config["nlist"]
            m = self.index_config["m"]  # Number of sub-vectors
            bits = self.index_config["bits"]  # Bits per sub-vector
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, bits, faiss.METRIC_INNER_PRODUCT)
            
        elif index_type == "HNSW":
            ef_construction = self.index_config["ef_construction"]
            self.index = faiss.IndexHNSWFlat(self.dimension, ef_construction)
            
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        if self.config["performance"]["use_gpu"]:
            self._enable_gpu()
        
        logger.info(f"Created new {index_type} index")
    
    def _enable_gpu(self):
        """Enable GPU acceleration for FAISS."""
        try:
            ngpus = faiss.get_num_gpus()
            if ngpus > 0:
                gpu_id = self.config["performance"]["gpu_id"]
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, gpu_id, self.index)
                logger.info(f"Enabled GPU acceleration on GPU {gpu_id}")
            else:
                logger.warning("No GPU available for FAISS")
        except Exception as e:
            logger.warning(f"Failed to enable GPU acceleration: {e}")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Embedding vectors (n_vectors, dimension)
            metadata: List of metadata dictionaries
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        faiss.normalize_L2(embeddings)
        
        if self.index.is_trained:
            self.index.add(embeddings)
        else:
            logger.info("Training FAISS index...")
            self.index.train(embeddings)
            self.index.add(embeddings)
        
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} embeddings to index")
        
        if len(self.metadata) % self.indexing_config["save_interval"] == 0:
            self.save_index()
    
    def search(self, query_embeddings: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar embeddings.
        
        Args:
            query_embeddings: Query embeddings (n_queries, dimension)
            k: Number of nearest neighbors to retrieve
            
        Returns:
            Tuple of (distances, indices)
        """
        if k is None:
            k = self.config["retrieval"]["k"]
        
        faiss.normalize_L2(query_embeddings)
        
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.index_config["nprobe"]
        
        distances, indices = self.index.search(query_embeddings, k)
        
        return distances, indices
    
    def get_metadata(self, indices: np.ndarray) -> List[Dict[str, Any]]:
        """
        Get metadata for given indices.
        
        Args:
            indices: Array of indices
            
        Returns:
            List of metadata dictionaries
        """
        metadata_list = []
        for idx in indices.flatten():
            if idx < len(self.metadata):
                metadata_list.append(self.metadata[idx])
            else:
                metadata_list.append({})
        
        return metadata_list
    
    def search_with_metadata(self, query_embeddings: np.ndarray, k: int = None) -> List[Dict[str, Any]]:
        """
        Search and return results with metadata.
        
        Args:
            query_embeddings: Query embeddings
            k: Number of nearest neighbors
            
        Returns:
            List of result dictionaries with metadata
        """
        distances, indices = self.search(query_embeddings, k)
        metadata_list = self.get_metadata(indices)
        
        results = []
        for i in range(len(query_embeddings)):
            query_results = []
            for j in range(len(indices[i])):
                if indices[i][j] < len(self.metadata):
                    result = {
                        "index": int(indices[i][j]),
                        "distance": float(distances[i][j]),
                        "similarity": float(distances[i][j]),  # For inner product
                        "metadata": metadata_list[i * len(indices[i]) + j]
                    }
                    query_results.append(result)
            results.append(query_results)
        
        return results
    
    def filter_search(self, query_embeddings: np.ndarray, filters: Dict[str, Any], k: int = None) -> List[Dict[str, Any]]:
        """
        Search with metadata filters.
        
        Args:
            query_embeddings: Query embeddings
            filters: Metadata filters
            k: Number of nearest neighbors
            
        Returns:
            Filtered results
        """
        
        all_results = self.search_with_metadata(query_embeddings, k * 2)  
        
        filtered_results = []
        for query_results in all_results:
            filtered_query_results = []
            for result in query_results:
                if self._matches_filters(result["metadata"], filters):
                    filtered_query_results.append(result)
                    if len(filtered_query_results) >= k:
                        break
            filtered_results.append(filtered_query_results)
        
        return filtered_results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if metadata matches filters.
        
        Args:
            metadata: Metadata dictionary
            filters: Filter dictionary
            
        Returns:
            True if matches, False otherwise
        """
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif isinstance(value, dict):
                if "min" in value and metadata[key] < value["min"]:
                    return False
                if "max" in value and metadata[key] > value["max"]:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def save_index(self):
        """Save index and metadata to disk."""
        try:
            index_path = self.indexing_config["index_path"]
            if hasattr(self.index, 'index'):
               
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_path)
            else:
                faiss.write_index(self.index, index_path)
            
            metadata_path = self.indexing_config["metadata_path"]
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            logger.info(f"Saved index with {len(self.metadata)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self, index_path: str, metadata_path: str):
        """
        Load index from specific paths.
        
        Args:
            index_path: Path to index file
            metadata_path: Path to metadata file
        """
        try:
            self.index = faiss.read_index(index_path)
            
            if self.config["performance"]["use_gpu"]:
                self._enable_gpu()
            
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Loaded index from {index_path} with {len(self.metadata)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Index statistics dictionary
        """
        stats = {
            "total_vectors": len(self.metadata),
            "dimension": self.dimension,
            "index_type": self.index_config["type"],
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            "ntotal": self.index.ntotal if hasattr(self.index, 'ntotal') else len(self.metadata)
        }
        
        if hasattr(self.index, 'nlist'):
            stats["nlist"] = self.index.nlist
        if hasattr(self.index, 'nprobe'):
            stats["nprobe"] = self.index.nprobe
        
        return stats
    
    def clear_index(self):
        """Clear all data from the index."""
        self._create_new_index()
        self.metadata = []
        logger.info("Cleared index")
    
    def remove_vectors(self, indices: List[int]):
        """
        Remove vectors from the index.
        
        Args:
            indices: List of indices to remove
        """
    
        logger.warning("Vector removal not fully implemented - consider rebuilding index")
        
        for idx in sorted(indices, reverse=True):
            if idx < len(self.metadata):
                del self.metadata[idx]
        
        self._create_new_index()
        logger.info("Index rebuilt after vector removal")
    
    def backup_index(self):
        """Create backup of the index."""
        backup_config = self.config["backup"]
        if not backup_config["enabled"]:
            return
        
        backup_path = backup_config["backup_path"]
        os.makedirs(backup_path, exist_ok=True)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_index_path = os.path.join(backup_path, f"index_{timestamp}.faiss")
        backup_metadata_path = os.path.join(backup_path, f"metadata_{timestamp}.json")
        
        if hasattr(self.index, 'index'):
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, backup_index_path)
        else:
            faiss.write_index(self.index, backup_index_path)
        
        with open(backup_metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self._cleanup_old_backups(backup_path, backup_config["max_backups"])
        
        logger.info(f"Created backup: {backup_index_path}")
    
    def _cleanup_old_backups(self, backup_path: str, max_backups: int):
        """Remove old backup files."""
        backup_files = []
        for file in os.listdir(backup_path):
            if file.startswith("index_") and file.endswith(".faiss"):
                file_path = os.path.join(backup_path, file)
                backup_files.append((file_path, os.path.getmtime(file_path)))
        
        backup_files.sort(key=lambda x: x[1])
        
        while len(backup_files) > max_backups:
            oldest_file, _ = backup_files.pop(0)
            try:
                os.remove(oldest_file)
               
                metadata_file = oldest_file.replace("index_", "metadata_").replace(".faiss", ".json")
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                logger.info(f"Removed old backup: {oldest_file}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {oldest_file}: {e}") 
                