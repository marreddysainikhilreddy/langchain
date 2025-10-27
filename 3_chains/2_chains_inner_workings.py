from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_ollama import ChatOllama



model = ChatOllama(model="llama3.2")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)


# create individual runnables (steps in the chain)
# A runnable lambda is just a wrapper that lets us create each task as a single reusable unit
# each runnable lambda takes an input, does some work with it (like filling in a prompt)
format_prompt = RunnableLambda(lambda x: prompt_template.format(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# create the runnable sequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

result = chain.invoke({"animal": "cat", "fact_count": 2})

print(result)