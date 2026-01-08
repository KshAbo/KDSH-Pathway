# import pathway as pw
# import openai
# from config import EMBEDDING_MODEL

# openai.api_key = None  # picked up from env


# def embed(text: str):
#     resp = openai.embeddings.create(
#         model=EMBEDDING_MODEL,
#         input=text
#     )
#     return resp.data[0].embedding


# class VectorStore:
#     def __init__(self):
#         self.schema = pw.Schema(
#             text=str,
#             embedding=list[float],
#         )

#         self.table = pw.Table.empty(schema=self.schema)

#     def add_documents(self, texts):
#         rows = [{"text": t, "embedding": embed(t)} for t in texts]
#         self.table += pw.debug.table_from_rows(rows)

#     def query(self, query_text, k=5):
#         query_embedding = embed(query_text)

#         scored = self.table.select(
#             text=self.table.text,
#             score=pw.math.cosine_similarity(
#                 self.table.embedding,
#                 query_embedding
#             )
#         )

#         topk = scored.sort(key=lambda r: -r.score).take(k)
#         return pw.debug.collect(topk)["text"]
import pathway as pw


class NovelChunk(pw.Schema):
    text: str
    novel_id: str


class VectorStore:
    def __init__(self):
        self.schema = NovelChunk
        self.table = None

    def add(self, text: str, novel_id: str):
        rows = [(text, novel_id)]
        new_table = pw.debug.table_from_rows(rows, schema=self.schema)

        if self.table is None:
            self.table = new_table
        else:
            self.table = pw.concat(self.table, new_table)

    def add_documents(self, chunks):
        """
        chunks: List[dict] with keys ['text', 'novel_id']
        """
        rows = [(c["text"], c["novel_id"]) for c in chunks]

        if not rows:
            return

        new_table = pw.debug.table_from_rows(rows, schema=self.schema)

        if self.table is None:
            self.table = new_table
        else:
            self.table = pw.concat(self.table, new_table)
