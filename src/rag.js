// ============================================================
// rag.js — Pipeline RAG complet (one-shot)
//
// RAG = Retrieval-Augmented Generation
// Le flow complet :
//   1. Question → Embedding (Ollama / nomic-embed-text)
//   2. Embedding → pgvector → Top K documents
//   3. Documents + Question → LLM (Ollama / llama3.2)
//   4. LLM → Reponse
//
// Usage : npm run rag -- "ta question"
// ============================================================

import pg from "pg";                                // Client PostgreSQL
import { OllamaEmbeddings } from "@langchain/ollama"; // Embeddings via Ollama
import {
  DB_CONFIG,                                        // Config PostgreSQL
  EMBEDDING_MODEL,                                  // nomic-embed-text
  CHAT_MODEL,                                       // llama3.2
  OLLAMA_BASE_URL,                                  // http://localhost:11434
} from "./config.js";

const { Client } = pg;

// --- Fonction RAG ---
// question = la question de l'utilisateur
// topK     = nombre de documents a recuperer (defaut: 3)
async function rag(question, topK = 3) {
  // Connexion PostgreSQL
  const client = new Client(DB_CONFIG);
  await client.connect();

  // Instance d'embeddings
  const embeddings = new OllamaEmbeddings({
    model: EMBEDDING_MODEL,                         // nomic-embed-text (768 dimensions)
    baseUrl: OLLAMA_BASE_URL,
  });

  console.log(`\nQuestion: "${question}"\n`);

  // --- Etape 1 : Generer l'embedding de la question ---
  // Le prefix "search_query: " est requis par nomic-embed-text
  console.log("1. Generating embedding...");
  const queryVector = await embeddings.embedQuery(`search_query: ${question}`); // Vecteur de 768 floats

  // --- Etape 2 : Chercher dans pgvector ---
  console.log("2. Searching pgvector...");
  const res = await client.query(
    `
    SELECT content, metadata, 1 - (embedding <=> $1::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> $1::vector
    LIMIT $2
    `,
    [JSON.stringify(queryVector), topK]
  );

  // Concatener les contenus des documents trouves pour former le contexte
  const context = res.rows.map((r) => r.content).join("\n\n");

  // Afficher les documents trouves avec leur score
  console.log(`   Found ${res.rows.length} relevant documents\n`);
  for (const row of res.rows) {
    const sim = (row.similarity * 100).toFixed(1);
    console.log(`   [${sim}%] ${row.content.slice(0, 80)}...`);
  }

  // --- Etape 3 : Construire le prompt pour le LLM ---
  // Le prompt contient :
  //   - Les instructions systeme (role, contraintes)
  //   - Le contexte (les documents pertinents trouves dans pgvector)
  //   - La question de l'utilisateur
  const prompt = `Tu es un assistant technique. Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si le contexte ne contient pas l'information, dis-le clairement.

Contexte:
${context}

Question: ${question}

Réponse:`;

  // --- Etape 4 : Appeler le LLM via l'API Ollama ---
  console.log("\n3. Calling LLM...\n");
  const llmRes = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
    method: "POST",                                 // Methode HTTP POST
    headers: { "Content-Type": "application/json" }, // On envoie du JSON
    body: JSON.stringify({
      model: CHAT_MODEL,                           // llama3.2
      prompt,                                       // Le prompt complet
      stream: false,                                // Attendre la reponse complete (pas de streaming token par token)
    }),
  });

  // Verifier que ca a marche
  if (!llmRes.ok) {
    throw new Error(`LLM call failed: ${llmRes.status} ${llmRes.statusText}`);
  }

  // Parser et afficher la reponse
  const llmData = await llmRes.json();              // { response: "...", ... }
  console.log("--- RESPONSE ---\n");
  console.log(llmData.response);                    // La reponse du LLM
  console.log("\n--- END ---");

  // Fermer la connexion PostgreSQL
  await client.end();
  return llmData.response;
}

// --- Point d'entree ---
const question =
  process.argv[2] || "Comment je fais pour réinitialiser mon mot de passe ?";
rag(question).catch((err) => {
  console.error("RAG failed:", err.message);
  process.exit(1);
});
