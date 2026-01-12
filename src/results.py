from indexing.novel_indexer import NovelIndexer
from reasoner import Reasoner

def main():
    indexer = NovelIndexer()

    # IMPORTANT:
    # If you already ingested via preprocess.py, indexer must ingest again
    # OR we can add serialization later— right now index rebuild is expected

    indexer.ingest("In search of the castaways", "./data/Books/In Search of the Castaways.txt")
    indexer.ingest("The Count of Monte Cristo", "./data/Books/The Count of Monte Cristo.txt")

    reasoner = Reasoner(
        indexer=indexer,
        book_name="The Count of Monte Cristo",
        top_k=3,
        out_path="out/results.jsonl"
    )

    reasoner.run()