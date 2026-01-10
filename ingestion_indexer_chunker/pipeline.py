import os
import pandas as pd
from tqdm.auto import tqdm
from config import Config, FileConfig
from engine import NovelIndexer

class CSVEnricher:
    def __init__(self, indexer: NovelIndexer):
        self.indexer = indexer
        self.books_indexed = set()
    
    def index_books_from_csv(self, csv_path: str):
        print(f"📖 Scanning books in: {os.path.basename(csv_path)}")
        
        try:
            df = pd.read_csv(csv_path)
            unique_books = df['book_name'].unique()
            
            for book_name in unique_books:
                if book_name in self.books_indexed:
                    continue
                
                novel_path = os.path.join(FileConfig.BOOKS_DIR, f"{book_name}.txt")
                self.indexer.ingest(book_name, novel_path)
                self.books_indexed.add(book_name)
                
        except FileNotFoundError:
            print(f"❌ CSV not found: {csv_path}")
    
    def enrich_csv(self, input_csv: str, output_csv: str, include_label: bool = True):
        if not os.path.exists(input_csv):
            print(f"Skipping {input_csv} (Not found)")
            return

        print(f"🚀 Enriching: {os.path.basename(input_csv)}")
        df = pd.read_csv(input_csv)
        
        chunk_columns = {f"chunk{i+1}": [] for i in range(Config.DEFAULT_TOP_K)}
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving Chunks"):
            book_name = row['book_name']
            query = f"{row['caption']}\n\n{row['content']}"
            
            chunks = self.indexer.retrieve_chunks(
                book_name=book_name,
                query=query,
                top_k=Config.DEFAULT_TOP_K
            )
            
            chunk_texts = [c.text for c in chunks]
            while len(chunk_texts) < Config.DEFAULT_TOP_K:
                chunk_texts.append("")
                
            for i, text in enumerate(chunk_texts[:Config.DEFAULT_TOP_K]):
                chunk_columns[f"chunk{i+1}"].append(text)
        
        for col, data in chunk_columns.items():
            df[col] = data
            
        # Reorder columns
        base_cols = ['id', 'book_name', 'char', 'caption', 'content']
        if include_label:
            base_cols.append('label')
        
        chunk_cols = [f'chunk{i+1}' for i in range(Config.DEFAULT_TOP_K)]
        df = df[base_cols + chunk_cols]
        
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved to {output_csv}")