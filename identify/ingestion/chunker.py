"""
Chunker — same shape as D:\\rag's Chunker (RecursiveCharacterTextSplitter,
chunk_size/overlap), fresh code. Slightly larger chunk size than the
reference (900 vs 500) since our documents are dense summaries, not raw
PDF text with a lot of whitespace/boilerplate to split away.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, text: str) -> list[str]:
        return self.splitter.split_text(text)
