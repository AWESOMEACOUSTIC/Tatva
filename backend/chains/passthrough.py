from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

model = ChatMistralAI(model = "mistral-small-2506")
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides code snippets in response to user queries."),
    ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides detailed explanations in response to user queries related to programming concepts."),
    ("human", "Explain the following code in programming with one dry run: \n{code}")
])

sequence_chain_1 = code_prompt | model | parser 
sequence_chain_2 = RunnableParallel(
    {
    "code" : RunnablePassthrough(),
    "explaination" : explain_prompt | model | parser
    }
)

chain = sequence_chain_1 | sequence_chain_2

result = chain.invoke({
    "topic": "write a code on reverse a linked list in Java?"
})

print(result["code"])
print(result["explaination"])