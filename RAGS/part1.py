import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Defining the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Inititializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    
    # Read the text content from the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # splitting the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n ----- Documents Chunks Information -----")
    print(f"Number of document chunks: {len(docs)}")

    # Creating embeddings
    print("\n-- creating embeddings --")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    ) 
    print("\n-- Finished creating embeddings --")

    # create the vector store and persist it automatically
    print("\n-- Creating vector store --")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("\n-- Finished creating vector store --")
else:
    print("Vector store already exists. No need to initialize")
