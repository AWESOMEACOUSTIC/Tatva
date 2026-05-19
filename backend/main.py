from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

load_dotenv()

# notes_path = Path(__file__).resolve().parent / "uploads" / "notes.txt"
# data = TextLoader(str(notes_path))
# docs = data.load()

pdf_path = Path(__file__).resolve().parent / "uploads" / "Ace.pdf"
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = splitter.split_documents(docs)


template = ChatPromptTemplate.from_messages([
    ("system", "you are a Ai that summarizes the text and gives the important points in the form of bullet points."),
    ("human", "Summarize the following text: {data}")
])

model = ChatMistralAI(model = "mistral-small-2506")
prompt = template.format_messages(data = chunks[3].page_content)

result = model.invoke(prompt)
print(result.content)

# model = ChatOpenRouter(model = "openai/gpt-4o-mini")
# result = model.invoke("how are you?")
# print(result.content)