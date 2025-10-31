import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    ) 

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "Where does Gandalf meet Frodo?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5},
)

relevant_docs = retriever.invoke(query)

print("\n-- Relevant Documents --")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")