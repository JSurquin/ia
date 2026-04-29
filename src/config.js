// ============================================================
// config.js — Configuration centralisee de l'application
//
// Toutes les constantes sont ici pour eviter de les dupliquer
// dans chaque fichier. Si tu changes de modele, de DB, ou
// de port, c'est ici que ca se passe.
// ============================================================

// --- Configuration PostgreSQL ---
// C'est la connexion a ta base de donnees PostgreSQL
// qui tourne dans Docker (docker-compose.yml)
export const DB_CONFIG = {
  user: "postgres",                // Utilisateur PostgreSQL (defini dans docker-compose.yml)
  password: "postgres",           // Mot de passe PostgreSQL (defini dans docker-compose.yml)
  host: "localhost",              // Host — localhost car Docker expose le port
  port: 5432,                    // Port PostgreSQL standard
  database: "ragdb",             // Nom de la base de donnees (creee automatiquement par Docker)
};

// --- Modele d'embedding ---
// nomic-embed-text transforme du texte en vecteur de 768 dimensions
// C'est ce vecteur qui est stocke dans pgvector et utilise pour la recherche semantique
export const EMBEDDING_MODEL = "nomic-embed-text";

// --- Dimension des embeddings ---
// nomic-embed-text genere des vecteurs de 768 nombres flottants
// La table PostgreSQL doit avoir une colonne VECTOR(768) qui correspond
// ATTENTION : si tu changes de modele, la dimension change aussi
// (ex: OpenAI = 1536, mxbai-embed-large = 1024)
export const EMBEDDING_DIM = 768;

// --- Modele LLM pour le chat ---
// llama3.2 = 2 milliards de parametres, ~2 Go
// C'est le plus leger de la famille Llama, ideal pour tourner en local
// Il recoit le contexte (documents trouves) + la question et genere la reponse
export const CHAT_MODEL = "llama3.2";

// --- URL du serveur Ollama ---
// Ollama tourne en local sur le port 11434
// On l'utilise pour 2 choses :
//   1. Generer les embeddings (via LangChain / OllamaEmbeddings)
//   2. Appeler le LLM (via l'API HTTP /api/generate)
export const OLLAMA_BASE_URL = "http://localhost:11434";
