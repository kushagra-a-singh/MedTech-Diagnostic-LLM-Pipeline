"""
FAISSIndex: a robust FAISS-based vector store compatible with your
vector_store_config.yaml and main.py usage.

Place this file at the repository root (next to main.py) or under src/ so
`from vector_store import FAISSIndex` resolves correctly.

Features:
- Loads existing FAISS index from config['indexing']['index_path'] if present
- Builds a new index from config['index'] parameters otherwise
- Persists index and metadata to disk
- Supports optional GPU transfer if `performance.use_gpu` is True
- Methods used by main.py: add_embeddings, search_with_metadata, get_index_stats

This implementation is defensive: it logs errors instead of raising in many
places so your API server will continue to run while the vector-store is
recovering or being built lazily.
"""

import json
import logging
import os
import threading
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Tuple

try:
    import faiss
except Exception:  # pragma: no cover - runtime resolution
    faiss = None

import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS-backed vector store manager.

    Expected config keys (based on your YAML):
      config['index'] -> index parameters (type, dimension, nlist, m, bits, ...)
      config['indexing']['index_path'] -> path to read/write faiss index
      config['indexing']['metadata_path'] -> json file storing metadata list
      config['performance'] -> gpu_id, use_gpu, num_threads

    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index_cfg = config.get("index", {})
        self.indexing_cfg = config.get("indexing", {})
        self.performance = config.get("performance", {})

        self.index_path = Path(self.indexing_cfg.get("index_path", "data/faiss_index"))
        self.metadata_path = Path(self.indexing_cfg.get("metadata_path", "data/faiss_metadata.json"))
        self.dimension = int(self.index_cfg.get("dimension", 768))

        self._lock = threading.Lock()

        self.index: Optional[Any] = None
        self.gpu_index = None
        self.metadata: List[Dict[str, Any]] = []
        self.next_id = 0

        # Try to load or build index
        try:
            self._init_index()
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.index = None

    def _init_index(self):
        """Load index from disk if present, otherwise build a new one based on config."""
        # Ensure data directory exists
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.metadata_path).parent.mkdir(parents=True, exist_ok=True)

        # Load metadata if exists
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
                self.next_id = len(self.metadata)
                logger.info(f"Loaded metadata with {len(self.metadata)} records")
            except Exception as e:
                logger.warning(f"Failed to load metadata file: {e}")
                self.metadata = []
                self.next_id = 0

        # If faiss is not available, keep index None
        if faiss is None:
            logger.warning("faiss library not available. Vector store functionality disabled.")
            return

        # If index file exists, load it
        if self.index_path.exists():
            try:
                logger.info(f"Loading FAISS index from {self.index_path}")
                self.index = faiss.read_index(str(self.index_path))
                logger.info("FAISS index loaded from disk")
                # Optionally move to GPU
                if self.performance.get("use_gpu", False):
                    self._maybe_move_to_gpu()
                return
            except Exception as e:
                logger.warning(f"Failed to read FAISS index file: {e}. Will try to rebuild.")

        # Build a new index from config
        logger.info("Building new FAISS index from config")
        idx_type = str(self.index_cfg.get("type", "Flat")).lower()

        if idx_type == "ivfflat" or idx_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = int(self.index_cfg.get("nlist", 100))
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
            logger.info(f"Created IndexIVFFlat with nlist={nlist}")
        elif idx_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("Created IndexFlatL2")
        else:
            logger.warning(f"Unknown index type '{idx_type}', defaulting to IndexFlatL2")
            self.index = faiss.IndexFlatL2(self.dimension)

        # For IVF indexes, they need training before adding vectors
        if isinstance(self.index, faiss.IndexIVF):
            logger.info("IndexIVF requires training. It will be trained on first add_embeddings batch.")

        # Save empty index to disk
        try:
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"Saved new FAISS index to {self.index_path}")
        except Exception as e:
            logger.warning(f"Failed to save new FAISS index: {e}")

    def _maybe_move_to_gpu(self):
        """Attempt to move CPU index to GPU if configured."""
        try:
            if faiss is None:
                return
            if not self.performance.get("use_gpu", False):
                return

            res = faiss.StandardGpuResources()
            gpu_id = int(self.performance.get("gpu_id", 0))
            logger.info(f"Moving FAISS index to GPU {gpu_id}")
            self.gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, self.index)
            logger.info("Index moved to GPU")
        except Exception as e:
            logger.warning(f"Failed to move index to GPU: {e}")
            self.gpu_index = None

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add embeddings (numpy array shape [N, D]) with corresponding metadata list.

        This method persists index and metadata to disk.
        """
        if faiss is None:
            logger.error("faiss not available — cannot add embeddings")
            return False

        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings to add")
            return False

        embeddings = np.array(embeddings).astype(np.float32)
        n, d = embeddings.shape
        if d != self.dimension:
            logger.error(f"Embedding dimension mismatch: expected {self.dimension}, got {d}")
            # Also check against actual index dimension if it exists
            if self.index:
                logger.error(f"Actual FAISS index dimension: {self.index.d}")
            return False

        with self._lock:
            try:
                # Train IVF if necessary
                if isinstance(self.index, faiss.IndexIVF) and not self.index.is_trained:
                    logger.info("Training IVF index on provided embeddings")
                    self.index.train(embeddings)

                # Add to index
                start_id = self.next_id
                self.index.add(embeddings)

                # Append metadata with assigned ids
                for i, md in enumerate(metadata):
                    entry = md.copy()
                    entry["id"] = start_id + i
                    self.metadata.append(entry)

                self.next_id += n

                # Persist index and metadata
                try:
                    faiss.write_index(self.index, str(self.index_path))
                except Exception as e:
                    logger.warning(f"Failed to write FAISS index to disk: {e}")

                try:
                    with open(self.metadata_path, "w") as f:
                        json.dump(self.metadata, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to write metadata file: {e}")

                logger.info(f"Added {n} embeddings to FAISS index")
                # Optionally move to GPU
                if self.performance.get("use_gpu", False):
                    self._maybe_move_to_gpu()

                return True

            except Exception as e:
                logger.error(f"Failed to add embeddings: {e}")
                logger.error(traceback.format_exc())
                return False

    def search_with_metadata(self, query_embeddings: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search the index and return results with metadata.

        Returns a list (per query) of dicts: { 'id', 'distance', 'metadata' }
        """
        if faiss is None:
            logger.error("faiss not available — cannot search")
            return []

        if query_embeddings is None or len(query_embeddings) == 0:
            logger.warning("Empty query embeddings")
            return []

        q = np.array(query_embeddings).astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        if q.shape[1] != self.dimension:
            logger.error(f"Query embedding dimension mismatch: expected {self.dimension}, got {q.shape[1]}")
            return []

        with self._lock:
            try:
                index_to_use = self.gpu_index if (self.gpu_index is not None) else self.index
                if index_to_use is None:
                    logger.error("FAISS index is not initialized")
                    return []

                distances, indices = index_to_use.search(q, k)

                results = []
                for qi in range(indices.shape[0]):
                    hits = []
                    for idx, dist in zip(indices[qi], distances[qi]):
                        if idx < 0:
                            continue
                        md = None
                        if idx < len(self.metadata):
                            md = self.metadata[idx]
                        hits.append({"id": int(idx), "distance": float(dist), "metadata": md})
                    results.append(hits)

                return results

            except Exception as e:
                logger.error(f"Search failed: {e}")
                logger.error(traceback.format_exc())
                return []

    def get_index_stats(self) -> Dict[str, Any]:
        """Return simple statistics about the index and metadata."""
        stats = {
            "index_exists": self.index_path.exists(),
            "metadata_count": len(self.metadata),
            "dimension": self.dimension,
            "use_gpu": bool(self.performance.get("use_gpu", False)),
        }

        try:
            if faiss is not None and self.index is not None:
                stats["ntotal"] = int(self.index.ntotal)
            else:
                stats["ntotal"] = 0
        except Exception:
            stats["ntotal"] = 0

        return stats


# if run as script, print status
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vector_store_config.yaml")
    args = parser.parse_args()

    # try loading YAML
    import yaml

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        vs = FAISSIndex(cfg.get("vector_store", cfg))
        print("Index stats:", vs.get_index_stats())
    else:
        print("Config not found:", args.config)
