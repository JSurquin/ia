# RAG avec PGVector + Ollama

App de Retrieval-Augmented Generation (RAG) en local.
PostgreSQL + pgvector pour le stockage vectoriel, Ollama pour les embeddings + LLM.
Zero API payante, tout tourne sur ta machine.

---

## Comment ca marche (schema)

```
┌─────────────────────────────────────────────────────┐
│                    PIPELINE RAG                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│   Ta question                                        │
│       │                                              │
│       ▼                                              │
│   Embedding (Ollama / nomic-embed-text)              │
│       │  → transforme le texte en vecteur (768 dim)  │
│       ▼                                              │
│   PostgreSQL + pgvector                              │
│       │  → recherche cosine similarity               │
│       │  → retourne les Top 3 documents              │
│       ▼                                              │
│   LLM (Ollama / llama3.2)                            │
│       │  → recoit contexte + question                │
│       │  → genere une reponse                        │
│       ▼                                              │
│   Reponse                                            │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Prerequis

| Outil    | Installation                        |
|----------|-------------------------------------|
| Docker   | OrbStack (`brew install orbstack`)  |
| Node.js  | >= 18 (`brew install node`)         |
| Ollama   | `brew install ollama`               |

---

## Setup rapide (tout-en-un)

```bash
# 1. Lancer Ollama (dans un terminal)
ollama serve

# 2. Dans un autre terminal, lancer le setup complet
cd ~/git/ia
npm run setup
```

Le setup va automatiquement :
1. Lancer PostgreSQL + pgvector dans Docker
2. Telecharger `nomic-embed-text` (embedding, 274 Mo)
3. Telecharger `llama3.2` (LLM, ~2 Go)
4. Creer la table `documents` + extension `vector` + index HNSW
5. Ingerer 10 documents exemple avec leurs embeddings

---

## Les 3 modes d'utilisation

### 1. Agent conversationnel (mode chat interactif)

```bash
npm run agent
```

C'est le mode principal. Tu discutes avec l'agent en boucle :
- Il cherche dans pgvector a chaque question
- Il garde l'historique de conversation
- Commandes speciales : `quit`, `clear` (effacer historique), `history`

### 2. RAG one-shot (une question → une reponse)

```bash
npm run rag -- "Comment reinitialiser mon mot de passe ?"
npm run rag -- "Comment exporter mes donnees ?"
npm run rag -- "Comment connecter Slack ?"
```

### 3. Recherche semantique pure (sans LLM)

```bash
npm run search -- "mot de passe"
npm run search -- "facturation"
npm run search -- "API authentication"
```

Utile pour debugger : voir quels documents remontent et avec quel score de similarite.

---

## Structure du projet

```
ia/
├── docker-compose.yml       ← PostgreSQL 16 + pgvector (Docker)
├── package.json             ← Scripts npm + dependances
├── README.md                ← Ce fichier
│
└── src/
    ├── config.js            ← Configuration centralisee (DB, modeles, URLs)
    ├── init-db.js           ← Cree extension vector + table documents + index HNSW
    ├── ingest.js            ← Ingere des documents (texte → embedding → PostgreSQL)
    ├── search.js            ← Recherche semantique pure (sans LLM)
    ├── rag.js               ← Pipeline RAG one-shot (search + LLM)
    ├── agent.js             ← Agent conversationnel interactif (chat loop)
    └── setup.js             ← Setup tout-en-un (Docker + modeles + DB + seed)
```

---

## Scripts npm disponibles

| Commande            | Description                                      |
|---------------------|--------------------------------------------------|
| `npm run setup`     | Setup complet (Docker + modeles + DB + seed)     |
| `npm run init-db`   | Creer la table + extension + index               |
| `npm run ingest`    | Ingerer les documents exemple                    |
| `npm run search`    | Recherche semantique (sans LLM)                  |
| `npm run rag`       | Pipeline RAG one-shot                            |
| `npm run agent`     | Agent conversationnel interactif                 |

---

## Setup manuel (etape par etape)

Si tu preferes faire chaque etape a la main :

```bash
# 1. Lancer PostgreSQL + pgvector
docker compose up -d

# 2. Telecharger les modeles Ollama
ollama pull nomic-embed-text    # Embedding (768 dimensions, 274 Mo)
ollama pull llama3.2            # LLM chat (2B params, ~2 Go)

# 3. Initialiser la base de donnees
npm run init-db

# 4. Ingerer les documents exemple
npm run ingest

# 5. Tester
npm run search -- "mot de passe"
npm run rag -- "Comment reinitialiser mon mot de passe ?"
npm run agent
```

---

## Technologies utilisees

| Composant      | Techno                        | Role                                    |
|----------------|-------------------------------|-----------------------------------------|
| Vector DB      | PostgreSQL 16 + pgvector      | Stockage et recherche vectorielle       |
| Embedding      | nomic-embed-text (768 dim)    | Transforme texte → vecteur              |
| LLM            | llama3.2 (2B params)          | Genere les reponses                     |
| Runtime        | Ollama                        | Fait tourner les modeles en local       |
| Backend        | Node.js + pg                  | Connexion PostgreSQL                    |
| Framework IA   | LangChain (@langchain/ollama) | Interface unifiee pour les embeddings   |

---

## Concepts cles

### Embedding
Un embedding c'est un vecteur (tableau de nombres) qui represente le "sens" d'un texte.
Deux textes qui parlent du meme sujet auront des vecteurs proches.

### Distance cosine
C'est la mesure de similarite entre deux vecteurs.
- 1.0 = identiques
- 0.0 = rien a voir
- pgvector utilise l'operateur `<=>` pour la distance cosine

### HNSW Index
Algorithme d'indexation pour accelerer la recherche de voisins les plus proches.
Sans index = scan sequentiel (lent). Avec HNSW = recherche rapide meme avec des millions de documents.

### RAG (Retrieval-Augmented Generation)
Au lieu de demander au LLM de tout savoir, on lui donne du contexte pertinent
trouve dans la base vectorielle. Le LLM repond uniquement a partir de ce contexte.

---

## Ajouter tes propres documents

Edite le tableau `SAMPLE_DOCS` dans `src/ingest.js` ou cree ton propre script :

```javascript
import pg from "pg";
import { OllamaEmbeddings } from "@langchain/ollama";
import { DB_CONFIG, EMBEDDING_MODEL, OLLAMA_BASE_URL } from "./config.js";

const client = new pg.Client(DB_CONFIG);
await client.connect();

const embeddings = new OllamaEmbeddings({
  model: EMBEDDING_MODEL,
  baseUrl: OLLAMA_BASE_URL,
});

// Ton document
const content = "Le texte de ton document ici";
const [vector] = await embeddings.embedDocuments([content]);

await client.query(
  "INSERT INTO documents (content, metadata, embedding) VALUES ($1, $2, $3)",
  [content, JSON.stringify({ category: "custom" }), JSON.stringify(vector)]
);

await client.end();
```

---

## Troubleshooting

| Probleme                          | Solution                                           |
|-----------------------------------|----------------------------------------------------|
| `docker compose up` fail          | Lancer OrbStack / Docker Desktop d'abord           |
| `ollama pull` fail                | Verifier que `ollama serve` tourne                  |
| `VECTOR type not found`           | Lancer `npm run init-db` (CREATE EXTENSION vector)  |
| Resultats de recherche nuls       | Verifier dimension embedding (768 pour nomic)       |
| Search casse apres changement    | Si tu changes de modele, re-ingerer tous les docs    |
