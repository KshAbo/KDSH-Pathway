import os
import re
import numpy as np
import pathway as pw
from pathway.stdlib.ml.index import KNNIndex
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

from config import Config
from data_types import Chunk, Evidence, PathwayChunkSchema, QuerySchema
from text_processing import NovelChunker

class PathwayVectorIndex:
    def __init__(self, embed_model: str = Config.EMBED_MODEL):
        self.embed_model = embed_model
        self.book_name = None
        self.chunks: List[Chunk] = []
        self.chunk_lookup: Dict[int, Chunk] = {}
        
        # Pathway Components
        self.pw_table: Optional[pw.Table] = None
        self.knn_index: Optional[KNNIndex] = None
        
        # Embedder
        self.embedder = SentenceTransformer(embed_model, device=Config.DEVICE)
        if Config.USE_FP16 and Config.DEVICE == "cuda":
            self.embedder.half()
        self.embedder.eval()
    
    def build(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")
        
        self.book_name = chunks[0].book_name
        self.chunks = chunks
        self.chunk_lookup = {c.chunk_id: c for c in chunks}
        
        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings_np = self.embedder.encode(
            texts,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=Config.DEVICE
        )
        
        # Prepare data for Pathway
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
        
        # Build Pathway KNN Index
        self.knn_index = KNNIndex(
            self.pw_table.embedding,
            self.pw_table,
            Config.EMBEDDING_DIM
        )
    
    def search(self, query: str, top_k: int) -> List[Evidence]:
        if self.knn_index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        if not self.chunks:
            return []
        
        top_k = min(top_k, len(self.chunks))
        
        query_emb = self.embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=Config.DEVICE
        )
        
        # NOTE: Using manual dot product here to return immediate results
        # in a script context, as Pathway KNN is optimized for streaming.
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
        
        evidences = []
        for rank, (chunk_id, score) in enumerate(top_results, 1):
            chunk = self.chunk_lookup[chunk_id]
            evidences.append(Evidence(chunk=chunk, score=score, rank=rank))
        
        return evidences

# the bbelow function seems to be shortened by GPT

class NovelIndexer:
    def __init__(self):
        self.chunker = NovelChunker()
        self.indices: Dict[str, PathwayVectorIndex] = {}
        print(f"✅ NovelIndexer Initialized ({Config.DEVICE})")
    
    def ingest(self, book_name: str, novel_path: str):
        if not os.path.exists(novel_path):
            print(f"❌ File not found: {novel_path}")
            return

        with open(novel_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunks = self.chunker.chunk(text, book_name)
        index = PathwayVectorIndex()
        index.build(chunks)
        
        self.indices[book_name] = index
        print(f"✅ Indexed '{book_name}' ({len(chunks)} chunks)")
    
    def retrieve_chunks(self, book_name: str, query: str, top_k: int = Config.DEFAULT_TOP_K) -> List[Chunk]:
        if book_name not in self.indices:
            return []
        evidences = self.indices[book_name].search(query, top_k)
        return [ev.chunk for ev in evidences]