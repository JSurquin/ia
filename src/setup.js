// ============================================================
// setup.js — Script de setup tout-en-un
//
// Lance tout dans l'ordre :
//   1. Demarre PostgreSQL + pgvector via Docker Compose
//   2. Telecharge les modeles Ollama (embedding + chat)
//   3. Attend que PostgreSQL soit pret
//   4. Initialise la base de donnees (table + index)
//   5. Ingere les documents exemple avec leurs embeddings
//
// Usage : npm run setup
// ============================================================

import { execSync } from "child_process";           // Pour executer des commandes shell depuis Node.js
import pg from "pg";                                // Client PostgreSQL
import { DB_CONFIG, EMBEDDING_MODEL, CHAT_MODEL } from "./config.js"; // Config centralisee

// --- Fonction utilitaire pour executer une commande ---
// cmd   = la commande shell a executer
// label = description affichee dans le terminal
function run(cmd, label) {
  console.log(`\n>> ${label}...`);
  try {
    execSync(cmd, { stdio: "inherit" });            // stdio: "inherit" = afficher la sortie en temps reel
    console.log(`   OK`);
  } catch (err) {
    console.error(`   FAILED: ${err.message}`);
    throw err;                                      // Remonter l'erreur pour arreter le setup
  }
}

// --- Attendre que PostgreSQL soit pret ---
// Apres le docker compose up, PostgreSQL met quelques secondes a demarrer
// On essaie de se connecter en boucle jusqu'a ce que ca marche
async function waitForPostgres(maxRetries = 20) {
  console.log("\n>> Waiting for PostgreSQL...");
  for (let i = 0; i < maxRetries; i++) {            // Essayer maxRetries fois
    try {
      const client = new pg.Client(DB_CONFIG);      // Tenter une connexion
      await client.connect();                       // Si ca marche, PostgreSQL est pret
      await client.end();                           // Fermer la connexion de test
      console.log("   PostgreSQL is ready!");
      return;                                       // Sortir de la fonction
    } catch {
      process.stdout.write(".");                    // Afficher un point de progression
      await new Promise((r) => setTimeout(r, 1500)); // Attendre 1.5 secondes avant de reessayer
    }
  }
  throw new Error("PostgreSQL not ready after retries"); // Timeout si toujours pas pret
}

async function setup() {
  console.log("=== RAG + PGVector Setup ===\n");

  // Etape 1 : Lancer PostgreSQL + pgvector dans Docker
  // docker compose up -d = lancer en arriere-plan (detached)
  run("docker compose up -d", "Starting PostgreSQL + pgvector (Docker)");

  // Etape 2 : Telecharger les modeles Ollama
  // ollama pull = telecharge le modele s'il n'est pas deja present
  run(`ollama pull ${EMBEDDING_MODEL}`, `Pulling embedding model: ${EMBEDDING_MODEL}`);
  run(`ollama pull ${CHAT_MODEL}`, `Pulling chat model: ${CHAT_MODEL}`);

  // Etape 3 : Attendre que PostgreSQL soit pret a accepter des connexions
  await waitForPostgres();

  // Etape 4 : Initialiser la base de donnees (extension vector + table + index)
  run("node src/init-db.js", "Initializing database");

  // Etape 5 : Ingerer les documents exemple
  run("node src/ingest.js", "Ingesting sample documents");

  // Done !
  console.log("\n=== Setup complete! ===");
  console.log("\nUsage:");
  console.log('  npm run search -- "your query"      # Recherche semantique');
  console.log('  npm run rag -- "your question"       # RAG one-shot');
  console.log('  npm run agent                        # Agent conversationnel');
}

// Lancer le setup et gerer les erreurs
setup().catch((err) => {
  console.error("\nSetup failed:", err.message);
  process.exit(1);
});
