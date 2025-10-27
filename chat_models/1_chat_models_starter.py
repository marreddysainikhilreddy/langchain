from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2-vision")

result = llm.invoke("what is the square root of 49")
print(result.content)