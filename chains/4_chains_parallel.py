from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2")

# 1️⃣ Main summary prompt
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human", "Provide a brief summary of the movie {movie_name}."),
    ]
)

# 2️⃣ Sub-analysis templates
plot_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human", "Analyze the plot: {plot}. What are its strengths and weaknesses?"),
    ]
)

character_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human", "Analyze the characters: {characters}. What are their strengths and weaknesses?"),
    ]
)

# 3️⃣ Helper function to merge outputs
def combine_verdicts(inputs):
    return f"Plot Analysis:\n{inputs['branches']['plot']}\n\nCharacter Analysis:\n{inputs['branches']['characters']}"

# 4️⃣ Proper branch chains
plot_branch_chain = (
    plot_template
    | model
    | StrOutputParser()
)

character_branch_chain = (
    character_template
    | model
    | StrOutputParser()
)

# 5️⃣ Combine everything
chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableLambda(lambda summary: {"plot": summary, "characters": summary})  # Feed same summary to both branches
    | RunnableParallel(branches={"plot": plot_branch_chain, "characters": character_branch_chain})
    | RunnableLambda(combine_verdicts)
)

# 6️⃣ Run
result = chain.invoke({"movie_name": "Inception"})
print(result)