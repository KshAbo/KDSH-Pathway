import torch

class Config:
    # System Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1024 if DEVICE == "cuda" else 32
    USE_FP16 = DEVICE == "cuda"
    
    # Text Processing
    CHUNK_SIZE = 450
    CHUNK_OVERLAP = 65
    
    # Models
    EMBED_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    # Retrieval
    DEFAULT_TOP_K = 10
    CHARACTER_SEARCH_MULTIPLIER = 4

class FileConfig:
    # Directories
    BOOKS_DIR = "../data/Books/"
    
    # Input Files
    TRAIN_CSV = "../data/train.csv"
    TEST_CSV = "../data/test.csv"
    
    # Output Files
    TRAIN_OUTPUT = "../intermediate/train_with_chunks.csv"
    TEST_OUTPUT = "../intermediate/test_with_chunks.csv"