from ingest import load_novel, chunk_text
from vector_store import VectorStore
from config import CHUNK_SIZE, CHUNK_OVERLAP

NOVEL_PATHS = {
    "The Count of Monte Cristo": "data/monte_cristo.txt",
    "In Search of the Castaways": "data/castaways.txt"
}

def build_novel_stores():
    stores = {}
    for book, path in NOVEL_PATHS.items():
        text = load_novel(path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        store = VectorStore()
        store.add_documents(chunks)
        stores[book] = store
    return stores
