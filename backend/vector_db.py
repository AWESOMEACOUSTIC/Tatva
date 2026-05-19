"""
load pdf
split the pdf into chunks 
create the embeddings for the chunks
store them in the vector database
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

pdf_path = Path(__file__).resolve().parent / "uploads" / "Ace.pdf"
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings(model = "mistral-embed-2312")
vectorStore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(Path(__file__).resolve().parent / "chroma_db"),
    collection_name="important_notes"
)