from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3.2")

template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"

## Converting python string prompt into the prompt that langchain understands 
## It creates a list with one human message
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({
    "tone": "energetic",
    "company": "samsung",
    "position": "AI Engineer",
    "skill": "AI"
})

result = llm.invoke(prompt)

# print(prompt)   


# 2: Prompt with System and Human Messages (Using Tuples)
messages = [
    ('system', "You are a comedian who tells jokes about {topic}."),
    ('human', "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)
