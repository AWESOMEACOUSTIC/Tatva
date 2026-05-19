from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

docs = [
    Document(page_content = "React is a JavaScript library for building user interfaces. It is maintained by Facebook and a community of individual developers and companies. React can be used as a base in the development of single-page or mobile applications.", metadata = {"source": "notes.txt", "page": 1}),
    Document(page_content= "Next is a framework that is used for building React applications. It is based on React and provides additional features for server-side rendering and routing.", metadata = {"source": "notes.txt", "page": 2}),
    Document(page_content= "Neural Network is used in machine learning and deep learning. It is a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input.", metadata = {"source": "notes.txt", "page": 3})
]

embedding_model = MistralAIEmbeddings(model = "mistral-embed-2312")

persist_dir = Path(__file__).resolve().parent.parent / "chroma_db"

vectorStore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory=str(persist_dir),
    collection_name = "notes"
)

results = vectorStore.similarity_search("What is React and why should developers learn it?", k = 2)

for r in results:
    print(r.page_content)
    print(r.metadata)

retriver = vectorStore.as_retriever()
docs = retriver.invoke("Explain the concept of Neural Networks in simple terms.", k = 2)

for d in docs:
    print(d.page_content)
    print(d.metadata)