from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_ollama import ChatOllama



model = ChatOllama(model="llama3.2")

animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {animal}."),
        ("human", "Tell me {count} facts."),
    ]
)


# Define a prompt template for translation to French
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human", "Translate the following text to {language}: {text}"),
    ]
)

# Define additional processing steps using RunnableLambda
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "french"})



# Create the combined chain using LangChain Expression Language (LCEL)

chain = animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser()

result = chain.invoke({"animal": "cat", "count": 2})

print(result)