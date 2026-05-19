from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from pathlib import Path

pdf_path = Path(__file__).resolve().parent / "Ace.pdf"
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

splitter = TokenTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10
)

chunks = splitter.split_documents(docs)
print(chunks[0].page_content)

#every chunk has the metadata and the page content. The metadata is the information about the document such as the source,