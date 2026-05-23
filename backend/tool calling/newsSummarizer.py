from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

search_tool = TavilySearchResults(max_results=3)

model = ChatMistralAI(model = "mistral-small-2506")

prompts = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that summarizes the search results and provides the important points in the form of bullet points.
    {news}
    """
)

chain = prompts | model | StrOutputParser()

news_result = search_tool.run("What are the latest news on Artificial Intelligence?")

result = chain.invoke({"news": news_result})

print(result)