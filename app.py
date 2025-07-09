# app.py - Interface Streamlit pour votre RAG

import streamlit as st
import os
from rag_engine import RAGEngine
import time
# app.py - Interface Streamlit pour votre RAG

# Configuration de la page
st.set_page_config(
    page_title="RAG Sénégal - Assistant Budgétaire",
    page_icon="🇸🇳",
    layout="wide"
)

# CSS pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #2E8B57;
    }
    .bot-message {
        background-color: #e8f5e8;
        border-left: 4px solid #228B22;
    }
    .source-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🇸🇳 RAG Sénégal - Assistant Budgétaire</h1>', unsafe_allow_html=True)

# Fonction pour initialiser le RAG
@st.cache_resource
def init_rag():
    """
    Initialise le RAG (mise en cache pour éviter de recharger)
    """
    with st.spinner(" Initialisation du moteur RAG..."):
        return RAGEngine()

# Fonction pour indexer le document
def index_document(rag_engine, pdf_path):
    """
    Indexe un document PDF
    """
    try:
        with st.spinner(" Indexation du document..."):
            num_chunks = rag_engine.index_document(pdf_path)
        st.success(f" Document indexé avec succès ! {num_chunks} chunks créés")
        return True
    except Exception as e:
        st.error(f" Erreur d'indexation : {str(e)}")
        return False

# Interface principale
def main():
    # Initialisation du RAG
    rag = init_rag()
    
    # Sidebar pour la gestion des documents
    st.sidebar.title(" Gestion des Documents")
    
    # Vérifier si des documents sont déjà indexés
    try:
        doc_count = rag.collection.count()
        if doc_count > 0:
            st.sidebar.success(f" {doc_count} chunks indexés")
        else:
            st.sidebar.warning(" Aucun document indexé")
    except:
        st.sidebar.error(" Erreur de connexion à la base")
    
    # Section d'indexation
    st.sidebar.subheader(" Indexer un document")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choisir un fichier PDF",
        type=['pdf'],
        help="Uploadez un document PDF à indexer"
    )
    
    if uploaded_file is not None:
        # Sauvegarder le fichier temporairement
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        if st.sidebar.button(" Indexer ce document"):
            success = index_document(rag, temp_path)
            if success:
                st.rerun()  # Recharger la page pour mettre à jour le compteur
        
        # Nettoyer le fichier temporaire
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Interface de chat principal
    st.subheader(" Assistant RAG")
    
    # Initialiser l'historique des conversations
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Afficher l'historique des conversations
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Afficher les sources si c'est une réponse du bot
            if message["role"] == "assistant" and "sources" in message:
                with st.expander(" Sources utilisées"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {i}</strong> (Similarité: {source['similarity']:.2f})<br>
                            <em>{source['text']}</em>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Zone de saisie pour les questions
    if prompt := st.chat_input("Posez votre question sur le budget du Sénégal..."):
        # Ajouter la question de l'utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Générer et afficher la réponse
        with st.chat_message("assistant"):
            with st.spinner(" Recherche et génération de la réponse..."):
                try:
                    result = rag.generate_answer(prompt)
                    
                    # Afficher la réponse
                    st.markdown(result["answer"])
                    
                    # Ajouter à l'historique avec les sources
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
                    
                    # Afficher les sources
                    if result["sources"]:
                        with st.expander(" Sources utilisées"):
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {i}</strong> (Similarité: {source['similarity']:.2f})<br>
                                    <em>{source['text']}</em>
                                </div>
                                """, unsafe_allow_html=True)
                    
                except Exception as e:
                    error_msg = f" Erreur : {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar avec des exemples de questions
    st.sidebar.subheader(" Questions d'exemple")
    example_questions = [
        "Quel est le montant des recettes fiscales ?",
        "Comment ont évolué les dépenses publiques ?",
        "Quels sont les principaux ministères du budget ?",
        "Quelle est la situation budgétaire du premier trimestre ?",
        "Quelles sont les recettes non fiscales ?"
    ]
    
    for question in example_questions:
        if st.sidebar.button(question, key=question):
            # Ajouter la question à l'historique et la traiter
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ** Statut technique :**
    - Modèle : Llama 3.2 (3B)
    - Embeddings : all-MiniLM-L6-v2
    - Base vectorielle : ChromaDB
    """)

if __name__ == "__main__":
    main()