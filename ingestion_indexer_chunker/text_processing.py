import nltk
from tqdm.auto import tqdm
from typing import List
from data_types import Chunk
from config import Config

# Ensure NLTK data exists
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

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