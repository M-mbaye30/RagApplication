import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.agents import Tool 

from rag_tool import search_rag_database
from web_search_tool import search_ministere_web

load_dotenv()

#Définition des outils pour l'agent principal
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
        description="""Effectue une recherche sur les sites officiels du gouvernement sénégalais, notamment finances.gouv.sn. 
        Utile pour trouver des liens, des informations générales, ou des mises à jour qui ne sont pas dans le document PDF. 
        Prend une chaîne de caractères en entrée.""",
    )
]

#Création du prompt de l'agent principal
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant expert sur les finances publiques du Sénégal. Réponds de manière précise et complète. Utilise uniquement les outils disponibles pour trouver les informations nécessaires. Si une question ne concerne pas un de tes outils, tu dois l'indiquer."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

#Création de l'Agent d'exécution
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_openai_functions_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#Exécution de l'orchestrateur
print("--- Requête 1 : Recherche dans le document PDF ---")
result_rag = agent_executor.invoke({"input": "Quel est le déficit budgétaire prévu pour 2026, selon le document de programmation budgétaire ?"})
print(f"\nRéponse : {result_rag['output']}")
print("\n" + "="*50 + "\n")

print("--- Requête 2 : Recherche sur le web ---")
result_web = agent_executor.invoke({"input": "Quel est le site web officiel du ministère des Finances au Sénégal ?"})
print(f"\nRéponse : {result_web['output']}")
print("\n" + "="*50 + "\n")

print("--- Requête 3 : Combinaison des deux ---")
result_combined = agent_executor.invoke({"input": "Comment le budget pour 2027 a-t-il été élaboré et où puis-je trouver le document officiel sur le site du ministère ?"})
print(f"\nRéponse : {result_combined['output']}")
print("\n" + "="*50 + "\n")

print("--- Requête 4 : Hors du périmètre ---")
result_irrelevant = agent_executor.invoke({"input": "Quelle est la capitale de la France ?"})
print(f"\nRéponse : {result_irrelevant['output']}")