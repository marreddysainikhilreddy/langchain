from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama



model = ChatOllama(model="llama3.2")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)

# content field from the models response is sent to the parser
chain = prompt_template | model | StrOutputParser()


# running the chain
result = chain.invoke({"animal": "cat", "fact_count": 2})

print(result)