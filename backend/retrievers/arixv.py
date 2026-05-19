import arxiv
from langchain_community.retrievers import ArxivRetriever

# arxiv.py v1 defaults to http; force https to avoid 301 errors.
arxiv.Client.query_url_format = "https://export.arxiv.org/api/query?{}"

retriever = ArxivRetriever(
    load_max_docs=3,
    load_all_available_meta=True
)

docs = retriever.invoke("quantum computing")

for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(f"Title: {doc.metadata.get('Title')}")
    print(f"Authors: {(doc.metadata.get('Authors'))}")
    print(f"Abstract: {doc.page_content[:800]}\n")