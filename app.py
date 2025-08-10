import streamlit as st
import os
import re
import time
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.agents import Tool
from rag_tool import search_rag_database
from web_search_tool import search_ministere_web

load_dotenv()

#Initialisation de l'Agent RAG
@st.cache_resource
def setup_agent():
    tools = [
        Tool(
            name="search_rag_database",
            func=search_rag_database,
            description="""Recherche des informations dans la base de données RAG sur le budget et les finances. 
            Utile pour répondre aux questions sur le contenu du document budgétaire. 
            Prend une chaîne de caractères en entrée.""",
        ),
        Tool(
            name="search_ministere_web",
            func=search_ministere_web,
            description="""Effectue une recherche sur les sites officiels du gouvernement sénégalais. 
            **Utiliser cet outil si l'information recherchée n'est pas trouvée dans la base de données interne (RAG).** Il est idéal pour trouver des chiffres à jour, des publications récentes, ou des documents qui ne sont pas dans le PDF de référence.""",
        )
    ]
    
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un assistant expert sur les finances publiques du Sénégal. Réponds de manière précise et complète.
        Utilise tes outils pour trouver les informations nécessaires. **Si le premier outil de recherche interne ne trouve pas l'information, tu dois obligatoirement essayer d'utiliser l'outil de recherche web pour trouver la réponse.**
        Si tu trouves des liens vers des sources pertinentes, intègre-les toujours dans la réponse."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_openai_functions_agent(llm, tools, agent_prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor = setup_agent()

#Fonctions utilitaires
def extract_links_from_text(text):
    """Extrait les URL brutes d'une chaîne de texte."""
    url_pattern = r'(https?://[^\s\)]+)'
    links = re.findall(url_pattern, text)
    unique_links = sorted(list(set(links)))
    return unique_links

def handle_query(user_query):
    """Fonction pour gérer l'exécution de la requête et la mise à jour de l'historique."""
    if user_query:
        # message de l'utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": user_query, "sources": [], "time": 0.0})
        
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Recherche de la meilleure réponse..."):
                try:
                    start_time = time.time()
                    response = agent_executor.invoke({"input": user_query})
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    response_content = response['output']
                    links = extract_links_from_text(response_content)
                    
                    st.markdown(response_content)
                    if links:
                        st.markdown("---")
                        st.subheader("Sources :")
                        for link in links:
                            st.markdown(f"- [{link}]({link})")
                    
                    st.markdown(f"<p style='color:green; font-weight:bold; font-size:12px; font-style:italic;'>Temps de traitement : {processing_time:.2f} secondes</p>", unsafe_allow_html=True)

                    st.session_state.messages.append({"role": "assistant", "content": response_content, "sources": links, "time": processing_time})
                    
                except Exception as e:
                    st.error(f"Une erreur est survenue : {e}")
        
        # Effacer la question présélectionnée de l'état de la session
        if 'preset_query' in st.session_state:
            del st.session_state.preset_query
            st.rerun()

#l'interface Streamlit
st.set_page_config(page_title="Assistant RAG Finances Publiques du Sénégal", layout="wide")

st.title("Assistant RAG Finances Publiques du Sénégal 🇸🇳")
st.markdown("Posez-moi une question sur le budget et la politique financière du Sénégal.")

# Init de l'historique de la conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# questions pré-sélectionnées
preset_questions = [
    "Quel est le déficit budgétaire prévu pour 2026 ?",
    "Quel est le site web officiel du ministère des Finances au Sénégal ?",
    "Comment le budget pour 2027 a-t-il été élaboré et où puis-je trouver le document officiel ?",
    "Quelle est la différence entre les recettes et les dépenses en 2026 ?",
    "Où puis-je trouver les publications officielles du ministère de l'Économie ?"
]

st.markdown("---")

# Boutons pour les questions pré-sélectionnées
st.markdown("##### 🚀 Ou essayez l'une de ces questions :")
cols = st.columns(len(preset_questions))
for i, question in enumerate(preset_questions):
    if cols[i].button(question, key=f"q{i}"):
        st.session_state.preset_query = question

# Affichage des messages existants
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["sources"]:
            st.markdown("---")
            st.subheader("Sources :")
            for source in message["sources"]:
                st.markdown(f"- [{source}]({source})")
        if message["time"]:
            st.markdown(f"<p style='color:green; font-weight:bold; font-size:12px; font-style:italic;'>Temps de traitement : {message['time']:.2f} secondes</p>", unsafe_allow_html=True)

# Lancement du traitement si une question est pré-sélectionnée
if 'preset_query' in st.session_state and st.session_state.preset_query:
    handle_query(st.session_state.preset_query)

# Champ de saisie pour l'utilisateur (après l'historique)
user_query = st.chat_input("Votre question :")

if user_query:
    handle_query(user_query)