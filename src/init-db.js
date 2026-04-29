// ============================================================
// init-db.js — Initialisation de la base de donnees
//
// Ce script fait 3 choses :
//   1. Active l'extension "vector" dans PostgreSQL (pgvector)
//   2. Cree la table "documents" avec une colonne VECTOR(768)
//   3. Cree un index HNSW pour accelerer les recherches
//
// Usage : npm run init-db
// ============================================================

import pg from "pg";                                // Client PostgreSQL pour Node.js
import { DB_CONFIG, EMBEDDING_DIM } from "./config.js"; // Config de connexion + dimension des vecteurs

const { Client } = pg;                             // On destructure la classe Client

async function initDB() {
  // --- Etape 1 : Connexion a PostgreSQL ---
  const client = new Client(DB_CONFIG);             // Creer un client avec la config (host, port, user, password, database)
  await client.connect();                           // Se connecter a la base
  console.log("Connected to PostgreSQL");

  // --- Etape 2 : Activer l'extension pgvector ---
  // Sans cette extension, PostgreSQL ne connait pas le type VECTOR
  // CREATE EXTENSION IF NOT EXISTS = ne plante pas si deja active
  await client.query("CREATE EXTENSION IF NOT EXISTS vector");
  console.log("Extension 'vector' enabled");

  // --- Etape 3 : Creer la table documents ---
  // id         = identifiant unique auto-incremente
  // content    = le texte du document (ce qu'on affiche a l'utilisateur)
  // metadata   = infos supplementaires en JSON (categorie, langue, etc.)
  // embedding  = le vecteur genere par nomic-embed-text (768 dimensions)
  await client.query(`
    CREATE TABLE IF NOT EXISTS documents (
      id SERIAL PRIMARY KEY,
      content TEXT NOT NULL,
      metadata JSONB DEFAULT '{}',
      embedding VECTOR(${EMBEDDING_DIM})
    )
  `);
  console.log(`Table 'documents' ready (embedding dim: ${EMBEDDING_DIM})`);

  // --- Etape 4 : Creer l'index HNSW ---
  // HNSW = Hierarchical Navigable Small World
  // C'est un algorithme d'indexation pour la recherche de voisins les plus proches
  // vector_cosine_ops = on utilise la distance cosine (la meilleure pour du texte)
  // Sans cet index, pgvector fait un scan sequentiel (lent sur beaucoup de documents)
  await client.query(`
    CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents
    USING hnsw (embedding vector_cosine_ops)
  `);
  console.log("HNSW index created for fast cosine search");

  // --- Fermer la connexion ---
  await client.end();                               // Toujours fermer proprement
  console.log("DB init complete!");
}

// Lancer l'init et gerer les erreurs
initDB().catch((err) => {
  console.error("DB init failed:", err.message);
  process.exit(1);                                  // Code de sortie 1 = erreur
});
