import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_community.vectorstores.utils import filter_complex_metadata

load_dotenv()

# Configuration 
pdf_path = "data/DPBEP-2026-2028.pdf"
persist_directory = "./chroma_db"

# Préparation du document 
loader = UnstructuredPDFLoader(pdf_path, mode="elements", strategy="fast", languages=["fra"])
documents = loader.load()

# Utilisation de l'utilitaire de nettoyage sur les documents
cleaned_documents = filter_complex_metadata(documents)

# Le découpage en chunks avec LangChain
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(cleaned_documents)

#Génération des embeddings et création de l'index
embeddings = OpenAIEmbeddings()

# Création et persistance de l'index dans ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Sauvegarde de l'index
vector_store.persist()

print(f"Index de vecteurs créé avec {len(chunks)} chunks et sauvegardé dans '{persist_directory}'.")
print("Le pipeline d'indexation est terminé avec succès.")