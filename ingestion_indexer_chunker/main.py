import os
from engine import NovelIndexer
from pipeline import CSVEnricher
from config import FileConfig

def ensure_directories():
    os.makedirs(FileConfig.BOOKS_DIR, exist_ok=True)
    # Ensure directory for outputs exists
    os.makedirs(os.path.dirname(FileConfig.TRAIN_OUTPUT), exist_ok=True)

def main():
    print("="*60)
    print("  PATHWAY NOVEL INDEXING SYSTEM")
    print("="*60)
    
    ensure_directories()
    
    # 1. Initialize Indexer
    indexer = NovelIndexer()
    enricher = CSVEnricher(indexer)
    
    # 2. Index Books & Enrich Train Data
    enricher.index_books_from_csv(FileConfig.TRAIN_CSV)
    enricher.enrich_csv(
        FileConfig.TRAIN_CSV, 
        FileConfig.TRAIN_OUTPUT, 
        include_label=True
    )
    
    # 3. Index Books & Enrich Test Data
    enricher.index_books_from_csv(FileConfig.TEST_CSV)
    enricher.enrich_csv(
        FileConfig.TEST_CSV, 
        FileConfig.TEST_OUTPUT, 
        include_label=False
    )
    
    print("\n✅ Processing Complete.")

if __name__ == "__main__":
    main()