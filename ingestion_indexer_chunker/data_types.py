from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import pathway as pw

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

# Pathway Schemas
class PathwayChunkSchema(pw.Schema):
    chunk_id: int
    text: str
    start_pos: int
    end_pos: int
    book_name: str
    embedding: list

class QuerySchema(pw.Schema):
    query_embedding: list