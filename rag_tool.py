from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Configuration
persist_directory = "./chroma_db"
embeddings = OpenAIEmbeddings()

# Chargement de la base de données vectorielle ChromaDB
vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# Création du "Retrieval Chain" pour la recherche
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def search_rag_database(query) -> str:
    if isinstance(query, dict) and "question" in query:
        query = query["question"]

    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return context

# Pour tester 
if __name__ == "__main__":
    test_query = "Quelle est la politique budgétaire pour 2026 ?"
    print(f"Recherche de : '{test_query}'...")
    result = search_rag_database(test_query)
    print("\n--- Résultat de l'outil RAG ---")
    print(result)