from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

pdf_path = Path(__file__).resolve().parent / "notes.pdf"
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

print(docs)