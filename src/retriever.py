# retriever.py

from typing import List
from indexing.novel_indexer import NovelIndexer, Chunk

class Retriever:
    def __init__(self, indexer: NovelIndexer):
        self.indexer = indexer

    def retrieve_chunks(self, book_name: str, query: str, top_k: int = 3) -> List[Chunk]:
        """
        Returns a list of Chunk objects (not evidence objects)
        """
        return self.indexer.retrieve_chunks(book_name, query, top_k)
