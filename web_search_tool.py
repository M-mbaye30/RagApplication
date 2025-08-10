from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

serper_tool = GoogleSerperAPIWrapper()

#Définition de l'outil pour l'agent
def search_ministere_web(query: str) -> str:
    """
    Effectue une recherche sur les sites officiels du gouvernement sénégalais.
    Les domaines cibles sont : finances.gouv.sn et vie-publique.sn.
    """
    # recherche pour ciblée 
    targeted_query = f"{query} site:finances.gouv.sn OR site:vie-publique.sn"
    
    # Exécution
    results = serper_tool.run(targeted_query)
    
    return results

#tester 
if __name__ == "__main__":
    test_query = "politique budgétaire"
    print(f"Recherche de : '{test_query}' sur le web...")
    result = search_ministere_web(test_query)
    print("\n--- Résultat de l'outil de recherche web ---")
    print(result)