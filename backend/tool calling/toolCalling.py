from dotenv import load_dotenv

load_dotenv()

from langchain import tools
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool

#1 creating a tool

@tool
def get_text_length(text:str) -> int:
    """Calculate the length of the given text"""
    return len(text)

model = ChatMistralAI(model = "mistral-small-2506")

#2 tool binding
model_bind = model.bind_tools([get_text_length])

results = model_bind.invoke("Use the get_text_length tool to calculate the length of the following text: 'Hello, how are you?'")
# the above code will send a response which will include that this tool can be called to get the length of the text.


if results.model_bind:
    bind_call = results.model_bind[0]

    tool_result = get_text_length.invoke(bind_call["args"])

    final_response = model_bind.invoke(f"The length of the text is {tool_result}.")
    print(final_response.content)