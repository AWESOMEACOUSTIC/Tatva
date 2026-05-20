from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("MISTRAL_API_KEY"):
    raise RuntimeError("MISTRAL_API_KEY is not set. Add it to backend/.env or your environment.")

#components
model = ChatMistralAI(model = "mistral-small-2506")
parser = StrOutputParser()

#Two Different Prompts
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in a short and concise manner"
)

long_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in a detailed manner with examples, architecture, and applications"
)

parallel_chain = RunnableParallel({
    "short": RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser,
    "long": RunnableLambda(lambda x: x["long"]) | long_prompt | model | parser
})

result = parallel_chain.invoke({
    "short" : {"topic": "Artificial Intelligence"},
    "long" : {"topic": "Internet of Things"}
})
print(result['short'])
print(result['long'])