from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

messages = [
    SystemMessage("You are an expert in social media content strategy"),
    HumanMessage("Give a short tip to create engaging posts on Instagram")
]



result = llm.invoke(messages)
print(result.content)