from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable
from dotenv import load_dotenv

load_dotenv()

# modèle de sortie
class SynthesisOutput(BaseModel):
    synthesized_answer: str = Field(description="Une réponse complète et synthétisée en français, basée sur le contexte fourni.")
    source_documents: list[str] = Field(description="Une liste de liens ou de noms de documents utilisés pour la synthèse.")

#prompt de synthèse
synthetizer_prompt = ChatPromptTemplate.from_messages([
    ("system", """Tu es un agent de synthèse. Ton rôle est de combiner les informations fournies pour créer une réponse unique, cohérente et précise.
     Ne réponds qu'avec le contenu fourni, sans ajouter de connaissances externes.
     Si une des sources d'information est vide ou ne contient pas de réponse, tu dois l'indiquer clairement dans ta réponse finale."""),
    ("human", """
    Synthétise les informations suivantes pour répondre à la question : '{question}'.
    
    Informations provenant du document PDF :
    {rag_result}
    
    Informations provenant du web :
    {web_result}
    
    Réponds de manière complète en combinant ces deux sources.
    """)
])

#l'agent de synthèse
synthetizer_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
synthetizer_agent = synthetizer_prompt | synthetizer_llm