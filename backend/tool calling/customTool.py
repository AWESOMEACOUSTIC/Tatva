from langchain.tools import tool

@tool
def get_greeting(name: str) -> str: #decorator for creating a tool
    """Generate a greeting message for the user"""  #docstring
    return f"Hello, {name}! Welcome to our platform."

result = get_greeting.invoke({"name" : "Aesh"})
print(result)