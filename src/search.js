// ============================================================
// search.js — Recherche semantique pure (sans LLM)
//
// Ce script prend une question, la transforme en vecteur,
// et cherche les documents les plus proches dans pgvector.
// Pas de LLM ici — juste la recherche vectorielle brute.
// Utile pour debug / verifier que les bons docs remontent.
//
// Usage : npm run search -- "ta requete"
// ============================================================

import pg from "pg";                                // Client PostgreSQL
import { OllamaEmbeddings } from "@langchain/ollama"; // Embeddings via Ollama
import { DB_CONFIG, EMBEDDING_MODEL, OLLAMA_BASE_URL } from "./config.js"; // Config centralisee

const { Client } = pg;                             // Destructure Client

// --- Fonction de recherche ---
// query = la question de l'utilisateur (texte)
// topK  = nombre de resultats a retourner (defaut: 3)
async function search(query, topK = 3) {
  // Connexion a PostgreSQL
  const client = new Client(DB_CONFIG);
  await client.connect();

  // Creer l'instance d'embeddings
  const embeddings = new OllamaEmbeddings({
    model: EMBEDDING_MODEL,                         // nomic-embed-text
    baseUrl: OLLAMA_BASE_URL,                       // http://localhost:11434
  });

  console.log(`\nQuery: "${query}"\n`);

  // --- Etape 1 : Transformer la question en vecteur ---
  // embedQuery() genere un seul embedding pour une seule question
  // Le prefix "search_query: " est requis par nomic-embed-text
  // pour que les vecteurs de requete soient dans le bon espace
  const queryVector = await embeddings.embedQuery(`search_query: ${query}`);

  // --- Etape 2 : Recherche dans pgvector ---
  // <=> = operateur de distance cosine dans pgvector
  //       0 = vecteurs identiques, 2 = vecteurs opposes
  // 1 - distance = similarite (1 = parfait, 0 = rien a voir)
  // ORDER BY embedding <=> $1 = trier par distance croissante (les plus proches d'abord)
  // LIMIT $2 = ne garder que les top K
  const res = await client.query(
    `
    SELECT content, metadata, 1 - (embedding <=> $1::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> $1::vector
    LIMIT $2
    `,
    [JSON.stringify(queryVector), topK]             // $1 = vecteur JSON, $2 = nombre de resultats
  );

  // --- Afficher les resultats ---
  console.log(`Top ${topK} results:\n`);
  for (const row of res.rows) {
    const sim = (row.similarity * 100).toFixed(1);  // Convertir en pourcentage
    console.log(`  [${sim}%] ${row.content}`);      // Afficher le contenu du document
    console.log(`         metadata: ${JSON.stringify(row.metadata)}\n`); // Afficher les metadonnees
  }

  // Fermer la connexion
  await client.end();
  return res.rows;                                  // Retourner les resultats
}

// --- Point d'entree ---
// process.argv[2] = premier argument passe en ligne de commande
// Si aucun argument, on utilise une question par defaut
const query = process.argv[2] || "comment réinitialiser mon mot de passe";
search(query).catch((err) => {
  console.error("Search failed:", err.message);
  process.exit(1);
});
