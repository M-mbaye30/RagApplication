# RAG Sénégal - Assistant Budgétaire

Une application de génération augmentée par récupération (RAG) spécialisée dans l'analyse des documents budgétaires officiels du Sénégal. Cette application permet de poser des questions en langage naturel sur les documents gouvernementaux et d'obtenir des réponses précises basées sur le contenu officiel.

## 📋 Description

Ce système RAG combine la puissance de Llama 3.2 avec une base de données vectorielle ChromaDB pour analyser intelligemment les documents budgétaires du Sénégal. L'application extrait, indexe et permet d'interroger le contenu des rapports PDF officiels via une interface web intuitive.

## ✨ Fonctionnalités principales

- **Extraction PDF intelligente** : Traitement automatique des documents PDF avec PyPDF2
- **Indexation vectorielle** : Conversion des documents en embeddings pour recherche sémantique
- **Chat intelligent** : Interface conversationnelle avec historique des échanges
- **Sources traçables** : Affichage des passages exacts utilisés pour générer chaque réponse
- **Upload de documents** : Ajout facile de nouveaux documents via l'interface
- **Exemples pré-définis** : Questions types pour découvrir les fonctionnalités
- **Persistance des données** : Sauvegarde automatique de l'index vectoriel

## 🛠️ Architecture technique

- **Frontend** : Streamlit avec interface de chat moderne
- **Backend RAG** : Moteur personnalisé avec ChromaDB
- **LLM** : Llama 3.2 (3B) via Ollama
- **Embeddings** : all-MiniLM-L6-v2 (SentenceTransformers)
- **Base vectorielle** : ChromaDB avec persistance locale
- **Traitement PDF** : PyPDF2 pour l'extraction de texte

## 📦 Installation

### Prérequis

1. **Python 3.8+**
2. **Ollama** installé et configuré
3. **Modèle Llama 3.2** téléchargé

### Installation du projet

1. Clonez le dépôt :
```bash
git clone https://github.com/M-mbaye30/RagApplication.git
cd RagApplication
```

2. Créez un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

### Configuration d'Ollama

1. Installez Ollama : [https://ollama.ai](https://ollama.ai)

2. Téléchargez le modèle Llama 3.2 :
```bash
ollama pull llama3.2:3b
```

3. Démarrez le serveur Ollama :
```bash
ollama serve
```

Le serveur doit tourner sur `http://localhost:11500`

## 🚀 Utilisation

### Lancement de l'application

```bash
streamlit run app.py
```

L'interface sera accessible sur `http://localhost:8501`

### Utilisation de l'interface

1. **Indexation de documents** :
   - Utilisez le sidebar pour uploader vos PDF
   - Cliquez sur "Indexer ce document"
   - Attendez la confirmation d'indexation

2. **Questionnement** :
   - Tapez votre question dans le chat
   - Obtenez une réponse basée sur les documents indexés
   - Consultez les sources utilisées dans l'onglet déroulant

3. **Exemples pré-définis** :
   - Utilisez les questions d'exemple dans le sidebar
   - Explorez les différents types de requêtes possibles

### Utilisation programmatique

```python
from rag_engine import RAGEngine

# Initialisation
rag = RAGEngine(model_name="llama3.2:3b")

# Indexation d'un document
rag.index_document("rapport_budget_t1_2025.pdf")

# Questionnement
result = rag.generate_answer("Quel est le montant des recettes fiscales ?")
print(result["answer"])
print(f"Sources: {len(result['sources'])}")
```

## 📁 Structure du projet

```
RagApplication/
│
├── app.py                    # Interface Streamlit
├── rag_engine.py            # Moteur RAG principal
├── requirements.txt         # Dépendances Python
├── vectorstore/             # Base ChromaDB (créée automatiquement)
│   ├── chroma.sqlite3
│   └── documents_senegal/
├── documents/               # Dossier pour vos PDF (optionnel)
└── README.md               # Ce fichier
```

## ⚙️ Configuration avancée

### Paramètres du RAG

Dans `rag_engine.py`, vous pouvez ajuster :

```python
class RAGEngine:
    def __init__(self, model_name="llama3.2:3b"):
        # Modèle d'embedding
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100):
        # Taille des chunks et chevauchement
        
    def search_similar_chunks(self, query: str, n_results: int = 5):
        # Nombre de chunks récupérés
```

### Personnalisation des prompts

Le prompt RAG dans `generate_answer()` peut être modifié pour :
- Changer le style de réponse
- Ajouter des contraintes spécifiques
- Modifier la langue de réponse

## 🎯 Exemples de questions

### Questions budgétaires
- "Quel est le montant total des recettes fiscales au premier trimestre 2025 ?"
- "Comment ont évolué les recettes du Sénégal ?"
- "Quels sont les principaux ministères mentionnés dans le budget ?"

### Questions d'analyse
- "Quelle est la situation budgétaire actuelle ?"
- "Quelles sont les principales dépenses publiques ?"
- "Comment se répartissent les recettes non fiscales ?"

## 🔧 Dépannage

### Erreurs courantes

**Ollama non accessible** :
```bash
# Vérifiez qu'Ollama tourne
curl http://localhost:11500/api/version

# Redémarrez si nécessaire
ollama serve
```

**Modèle manquant** :
```bash
# Téléchargez le modèle
ollama pull llama3.2:3b

# Vérifiez les modèles installés
ollama list
```

**Erreur de mémoire** :
- Réduisez `chunk_size` dans `rag_engine.py`
- Limitez `n_results` dans les recherches
- Utilisez un modèle plus petit : `llama3.2:1b`

### Performance

**Améliorer la vitesse** :
- Utilisez un SSD pour `vectorstore/`
- Augmentez la RAM disponible
- Optimisez `chunk_size` selon vos documents

**Améliorer la qualité** :
- Nettoyez vos PDF avant indexation
- Ajustez les paramètres de chunking
- Expérimentez avec différents modèles d'embedding

## 📊 Métriques

- **Temps d'indexation** : ~30 secondes par PDF de 50 pages
- **Temps de réponse** : 3-8 secondes selon la complexité
- **Précision** : Score de similarité affiché pour chaque source
- **Capacité** : Limité par l'espace disque et la RAM

## 🤝 Contribution

1. Fork le projet
2. Créez votre branche (`git checkout -b feature/amelioration`)
3. Committez vos changements (`git commit -m 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrez une Pull Request

## 📄 Licence
