from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2")

# Solving a case where we run an e-commerce website and users can
# leave feedbacks on the products that they have bought and depending on 
# the type of feedback we get from the user, the response from the
# customer support agent would vary.

# defining the prompt for different feedback Types
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        ('human', 'Generate a thank you note for this positive feedback: {feedback}')
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        ('human', 'Generate a response addressing this negative feedback: {feedback}.')
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant'),
        ('human', 'Generate a request for more details for this neutral feedback: {feedback}')
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant'),
        ('human', 'Generate a message to escalate this feedback to a human agent: {feedback}.')
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        ('human', 'Classify the sentiment of this feedback as positive, negative, neutral or escalate: {feedback}.'),
    ]
)

# defining runnable branch for handling feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser() # positive feedback chain
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser() # negative feedback chain
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser() # neutral feedback chain
    ),
    escalate_feedback_template | model | StrOutputParser()
)

# creating the classification chain
classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches

review = "I’m disappointed with my experience using this product. The build quality feels much cheaper than advertised, and several features don’t work as described. The setup process was confusing, and the instructions lacked clarity. Performance is inconsistent — it often freezes or crashes during normal use. For the price, I expected something far more reliable and better supported. I wouldn’t recommend this product until these issues are addressed."

result = chain.invoke({"feedback": review})
print(result)