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


### Configuration d'Ollama

1. Installez Ollama : [https://ollama.ai](https://ollama.ai)

2. Téléchargez le modèle Llama 3.2 :
```bash
ollama pull llama3.2:3b
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

### Personnalisation des prompts
## 🎯 Exemples de questions
### Questions budgétaires
- "Quel est le montant total des recettes fiscales au premier trimestre 2025 ?"
- "Comment ont évolué les recettes du Sénégal ?"
- "Quels sont les principaux ministères mentionnés dans le budget ?"
- 
### Performance

## 📊 Métriques

## 🤝 Contribution

## 📄 Licence
