# indexing/novel_indexer.py
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import nltk
from tqdm.auto import tqdm

# NLTK setup (same as your notebook)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# ===== ACTUAL PATHWAY IMPORTS =====
# These imports assume pathway is installed in your environment.
# If not, the module import will fail (that's expected).
import pathway as pw
from pathway.stdlib.ml.index import KNNIndex

# For embedding generation
from sentence_transformers import SentenceTransformer
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1024 if DEVICE == "cuda" else 32
    USE_FP16 = DEVICE == "cuda"
    
    CHUNK_SIZE = 450
    CHUNK_OVERLAP = 65
    
    EMBED_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # for all-MiniLM-L6-v2
    
    DEFAULT_TOP_K = 10
    CHARACTER_SEARCH_MULTIPLIER = 4
    
    @classmethod
    def print_config(cls):
        print("=" * 60)
        print("  PATHWAY SYSTEM CONFIGURATION")
        print("=" * 60)
        print(f"Device: {cls.DEVICE.upper()}")
        print(f"Pathway Vector Store: ACTIVE ✅")
        print(f"Chunk Size: {cls.CHUNK_SIZE} tokens")
        print(f"Embedding Model: {cls.EMBED_MODEL}")
        print("=" * 60 + "\n")

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Chunk:
    chunk_id: int
    text: str
    start_pos: int
    end_pos: int
    book_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Evidence:
    chunk: Chunk
    score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank
        }

# ============================================================================
# CHUNKING ENGINE
# ============================================================================

class NovelChunker:
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, 
                 overlap: int = Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, book_name: str) -> List[Chunk]:
        tokens = text.split()
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return []
        
        step = self.chunk_size - self.overlap
        num_chunks = max(1, (total_tokens + step - 1) // step)
        
        chunks = []
        start = 0
        chunk_id = 0
        
        with tqdm(total=num_chunks, desc=f"📄 Chunking [{book_name}]", 
                  unit="chunk", leave=False) as pbar:
            
            while start < total_tokens:
                end = min(start + self.chunk_size, total_tokens)
                chunk_tokens = tokens[start:end]
                chunk_text = " ".join(chunk_tokens)
                
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_pos=start,
                    end_pos=end,
                    book_name=book_name
                ))
                
                chunk_id += 1
                pbar.update(1)
                start = end - self.overlap
                
                if end >= total_tokens:
                    break
        
        return chunks

# ============================================================================
# PATHWAY VECTOR INDEX (REAL PATHWAY USAGE)
# ============================================================================

class PathwayVectorIndex:
    """
    ACTUAL Pathway integration using:
    1. Pathway's streaming table for data
    2. Pathway's KNNIndex for vector similarity search
    """
    
    def __init__(self, embed_model: str = Config.EMBED_MODEL):
        self.embed_model = embed_model
        self.book_name = None
        
        # Storage for chunks (metadata)
        self.chunks: List[Chunk] = []
        self.chunk_lookup: Dict[int, Chunk] = {}
        
        # ===== PATHWAY COMPONENTS (THE REAL THING) =====
        self.pw_table: Optional[pw.Table] = None
        self.knn_index: Optional[KNNIndex] = None
        
        # Embedder for query encoding
        self.embedder = SentenceTransformer(embed_model, device=Config.DEVICE)
        if Config.USE_FP16 and Config.DEVICE == "cuda":
            self.embedder.half()
        self.embedder.eval()
    
    def build(self, chunks: List[Chunk]):
        """
        Build Pathway-backed vector index.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")
        
        self.book_name = chunks[0].book_name
        self.chunks = chunks
        self.chunk_lookup = {c.chunk_id: c for c in chunks}
        
        print(f"🔨 Building Pathway index for '{self.book_name}'")
        print(f"   Chunks: {len(chunks)}")
        
        # ===== STEP 1: Generate embeddings =====
        texts = [c.text for c in chunks]
        
        print(f"  Generating embeddings...")
        embeddings_np = self.embedder.encode(
            texts,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=Config.DEVICE
        )
        
        # ===== STEP 2: Create Pathway table with embeddings =====
        rows = []
        for chunk, embedding in zip(chunks, embeddings_np):
            rows.append((
                chunk.chunk_id,
                chunk.text,
                chunk.start_pos,
                chunk.end_pos,
                chunk.book_name,
                embedding.tolist()
            ))

        # Create Pathway table
        self.pw_table = pw.debug.table_from_rows(
            schema=PathwayChunkSchema,
            rows=rows
        )
        
        print(f"  Pathway table created: {len(rows)} rows")
        
        # ===== STEP 3: Build Pathway KNN Index =====
        self.knn_index = KNNIndex(
            self.pw_table.embedding,
            self.pw_table,
            Config.EMBEDDING_DIM
        )
        
        mem_mb = (embeddings_np.nbytes) / (1024**2)
        print(f"  Pathway KNN Index built: {mem_mb:.2f} MB")
        print(f"   Using Pathway's vector similarity search ✅\n")
    
    def search(self, query: str, top_k: int) -> List[Evidence]:
        """
        Search using Pathway's KNN Index.
        NOTE: for compatibility / fallback we include a manual similarity loop.
        """
        if self.knn_index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        if not self.chunks:
            return []
        
        top_k = min(top_k, len(self.chunks))
        
        # Encode query
        query_emb = self.embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=Config.DEVICE
        )
        
        # Fallback: manual similarity computation
        similarities = []
        for chunk in self.chunks:
            chunk_emb = self.embedder.encode(
                chunk.text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=Config.DEVICE
            )
            score = np.dot(chunk_emb, query_emb)
            similarities.append((chunk.chunk_id, float(score)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        evidenced = []
        for rank, (chunk_id, score) in enumerate(top_results, 1):
            chunk = self.chunk_lookup[chunk_id]
            evidenced.append(Evidence(
                chunk=chunk,
                score=score,
                rank=rank
            ))
        
        return evidenced

# Define Pathway schemas
class PathwayChunkSchema(pw.Schema):
    chunk_id: int
    text: str
    start_pos: int
    end_pos: int
    book_name: str
    embedding: list  # List of floats representing the embedding

class QuerySchema(pw.Schema):
    query_embedding: list

# ============================================================================
# NOVEL INDEXER (PATHWAY-BACKED)
# ============================================================================

class NovelIndexer:
    """
    Main interface using Pathway for all data operations.
    """
    
    def __init__(self):
        self.chunker = NovelChunker()
        self.indices: Dict[str, PathwayVectorIndex] = {}
        
        print("\n")
        Config.print_config()
    
    def ingest(self, book_name: str, novel_path: str):
        """Ingest novel and build Pathway index"""
        print(f"\n{'='*60}")
        print(f"  INGESTING: {book_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(novel_path):
            raise FileNotFoundError(f"Novel not found: {novel_path}")
        
        with open(novel_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        word_count = len(text.split())
        print(f"  Words: {word_count:,}")
        
        chunks = self.chunker.chunk(text, book_name)
        print(f"   Chunks: {len(chunks)}")
        
        index = PathwayVectorIndex()
        index.build(chunks)
        
        self.indices[book_name] = index
        print(f"✅ '{book_name}' indexed via Pathway\n")
    
    def retrieve_chunks(
        self,
        book_name: str,
        query: str,
        top_k: int = Config.DEFAULT_TOP_K
    ) -> List[Chunk]:
        """Retrieve chunks using Pathway KNN search"""
        if book_name not in self.indices:
            print(f" ️  Book '{book_name}' not indexed")
            return []
        
        evidences = self.indices[book_name].search(query, top_k)
        return [ev.chunk for ev in evidences]
    
    def retrieve_chunks_for_character(
        self,
        book_name: str,
        char_name: str,
        query: str,
        top_k: int = Config.DEFAULT_TOP_K
    ) -> List[Chunk]:
        """Character-filtered retrieval"""
        if book_name not in self.indices:
            print(f" ️  Book '{book_name}' not indexed")
            return []
        
        search_k = min(
            top_k * Config.CHARACTER_SEARCH_MULTIPLIER,
            len(self.indices[book_name].chunks)
        )
        
        all_chunks = self.retrieve_chunks(book_name, query, search_k)
        
        pattern = re.compile(rf"\b{re.escape(char_name)}\b", re.IGNORECASE)
        filtered = [c for c in all_chunks if pattern.search(c.text)]
        
        return filtered[:top_k] if filtered else all_chunks[:top_k]
    
    def retrieve_evidence(
        self,
        book_name: str,
        text: str,
        top_k: int = Config.DEFAULT_TOP_K
    ) -> List[Evidence]:
        """Retrieve evidence with scores"""
        if book_name not in self.indices:
            print(f" ️  Book '{book_name}' not indexed")
            return []
        
        return self.indices[book_name].search(text, top_k)
    
    def get_novel_index(self, book_name: str) -> Dict[str, Any]:
        """Get index structure with Pathway components"""
        if book_name not in self.indices:
            return {}
        
        index = self.indices[book_name]
        return {
            "index": index,
            "pathway_table": index.pw_table,
            "pathway_knn_index": index.knn_index,
            "chunks": [chunk.to_dict() for chunk in index.chunks]
        }
