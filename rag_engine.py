# rag_engine.py 

import os
from typing import List
import chromadb
import PyPDF2
from sentence_transformers import SentenceTransformer
import requests
import json


class RAGEngine:
    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        
        # 1. Modèle d'embedding (pour transformer le texte en vecteurs)
        print("Chargement du modèle d'embedding...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Base vectorielle Chroma
        print("Initialisation de la base vectorielle...")
        self.chroma_client = chromadb.PersistentClient(path="./vectorstore")
        
        # 3. Collection pour stocker nos documents
        try:
            self.collection = self.chroma_client.create_collection(
                name="documents_senegal",
                metadata={"description": "Documents officiels du Sénégal"}
            )
        except:
            # Si la collection existe déjà, on la récupère
            self.collection = self.chroma_client.get_collection("documents_senegal")
        
        print("Moteur RAG initialisé !")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        print(f"Extraction du texte depuis {pdf_path}...")
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
        
        print(f" Texte extrait : {len(text)} caractères")
        return text

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:

        print(f" Découpage du texte en chunks de {chunk_size} caractères...")
        
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk.strip()) > 50:  
                chunks.append(chunk.strip())
        
        print(f" {len(chunks)} chunks créés")
        return chunks

    def index_document(self, pdf_path: str):
        
        print(f" Indexation du document : {pdf_path}")
        
        # Étape 1 : Extraction
        text = self.extract_text_from_pdf(pdf_path)
        
        # Étape 2 : Découpage
        chunks = self.chunk_text(text)
        
        # Étape 3 & 4 : Embeddings + Stockage
        print("Création des embeddings et stockage...")
        
        # Préparer les données pour ChromaDB
        ids = []           # IDs uniques pour chaque chunk
        embeddings = []    # Vecteurs numériques
        documents = []     # Textes originaux
        metadatas = []     # Métadonnées (infos sur le chunk)
        
        for i, chunk in enumerate(chunks):
            # ID unique
            chunk_id = f"{os.path.basename(pdf_path)}_chunk_{i}"
            ids.append(chunk_id)
            
            # Convertir le chunk en vecteur numérique
            embedding = self.embedding_model.encode(chunk).tolist()
            embeddings.append(embedding)
            
            # Texte original
            documents.append(chunk)
            
            # Métadonnées
            metadatas.append({
                "source": pdf_path,
                "chunk_index": i,
                "length": len(chunk)
            })
        
        # Stocker dans ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Document indexé ! {len(chunks)} chunks ajoutés à la base vectorielle")
        return len(chunks)

    def search_similar_chunks(self, query: str, n_results: int = 5) -> List[dict]:
        
        print(f" Recherche pour : '{query}'")
        
        # Convertir la question en vecteur
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Rechercher les chunks similaires
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Formater les résultats
        similar_chunks = []
        for i in range(len(results['documents'][0])):
            similar_chunks.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        print(f"{len(similar_chunks)} chunks trouvés")
        return similar_chunks

    def query_ollama(self, prompt: str) -> str:
      
        url = "http://localhost:11500/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False  # Réponse complète d'un coup
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Erreur dans la génération")
        
        except requests.exceptions.ConnectionError:
            return "Erreur : Ollama n'est pas démarré. Lancez 'ollama serve' dans un terminal."
        except Exception as e:
            return f"Erreur Ollama : {str(e)}"

    def generate_answer(self, question: str, max_chunks: int = 1) -> dict:
        
        print(f"\n Génération de réponse pour : '{question}'")
        
        # Étape 1 : Recherche vectorielle
        similar_chunks = self.search_similar_chunks(question, n_results=max_chunks)
        
        if not similar_chunks:
            return {
                "answer": "Aucune information trouvée dans les documents indexés.",
                "sources": [],
                "context_used": ""
            }
        
        # Étape 2 : Construire le contexte
        context = "\n\n".join([chunk['text'] for chunk in similar_chunks])
        
        # Étape 3 : Créer le prompt RAG
        prompt = f"""Tu es un assistant spécialisé dans l'analyse des documents officiels du Sénégal.

CONTEXTE (extrait des documents officiels) :
{context}

QUESTION : {question}

INSTRUCTIONS :
- Réponds uniquement basé sur le contexte fourni
- Si l'information n'est pas dans le contexte, dis-le clairement
- Sois précis et factuel
- Utilise les chiffres exacts du document
- Réponds en français

RÉPONSE :"""

        # Étape 4 : Générer la réponse
        print(" Génération avec Ollama...")
        answer = self.query_ollama(prompt)
        
        # Étape 5 : Préparer les sources
        sources = []
        for chunk in similar_chunks:
            sources.append({
                "text": chunk['text'][:200] + "...",
                "source": chunk['metadata']['source'],
                "chunk_index": chunk['metadata']['chunk_index'],
                "similarity": 1 - chunk['distance']  # Convertir distance en similarité
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": context[:500] + "..." if len(context) > 500 else context
        }



# Test complet du RAG avec génération de réponses

if __name__ == "__main__":
    # Test complet
    rag = RAGEngine()
    
    # 1. Indexer votre document (seulement si pas déjà fait)
    print("\n=== ÉTAPE 1 : VÉRIFICATION INDEX ===")
    pdf_path = "..\..\..\rapport_budget_t1_2025.pdf"

    if os.path.exists(pdf_path):
        num_chunks = rag.index_document(pdf_path)
        print(f"Document indexé avec {num_chunks} chunks")
    else:
        print(f" PDF non trouvé à l'emplacement : {pdf_path}")
    
    # 2. Test du RAG complet
    print("\n=== ÉTAPE 2 : TEST RAG COMPLET ===")
    questions_test = [
        "Quel est le montant total des recettes fiscales au premier trimestre 2025 ?",
        "Comment ont évolué les recettes du Sénégal ?",
        "Quels sont les principaux ministères mentionnés dans le budget ?"
    ]
    
    for question in questions_test:
        print(f"\n" + "="*60)
        print(f"QUESTION : {question}")
        print("="*60)
        
        result = rag.generate_answer(question, max_chunks=1)
        
        print(f"RÉPONSE :")
        print(result['answer'])
        
        print(f"\n SOURCES ({len(result['sources'])}) :")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. Similarité: {source['similarity']:.2f}")
            print(f"     Extrait: {source['text']}")
            print()
    
    print("\n Tests RAG complets terminés !")
    print("\n PROCHAINE ÉTAPE : Interface Streamlit pour une utilisation facile !")