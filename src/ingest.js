// ============================================================
// ingest.js — Ingestion de documents dans pgvector
//
// Ce script prend des documents texte, genere leur embedding
// via Ollama (nomic-embed-text), et les stocke dans PostgreSQL
// avec leur vecteur. C'est l'etape "indexation" du RAG.
//
// Usage : npm run ingest
// ============================================================

import pg from "pg";                                // Client PostgreSQL
import { OllamaEmbeddings } from "@langchain/ollama"; // Classe LangChain pour les embeddings Ollama
import { DB_CONFIG, EMBEDDING_MODEL, OLLAMA_BASE_URL } from "./config.js"; // Config centralisee

const { Client } = pg;                             // Destructure Client depuis pg

// --- Documents exemple ---
// Chaque document a :
//   content  = le texte qui sera indexe et recherchable
//   metadata = des infos supplementaires (categorie, langue)
//
// En production, ces documents viendraient d'un fichier, d'une API,
// d'un scraping, etc. Ici c'est du sample pour tester.
const SAMPLE_DOCS = [
  {
    content:
      "Pour réinitialiser votre mot de passe, allez dans Paramètres > Sécurité > Réinitialiser le mot de passe. Un email de confirmation vous sera envoyé.",
    metadata: { category: "auth", lang: "fr" },     // Categorie auth, en francais
  },
  {
    content:
      "To reset your password, go to Settings > Security > Reset Password. A confirmation email will be sent to you.",
    metadata: { category: "auth", lang: "en" },     // Meme doc en anglais — la recherche semantique les trouvera tous les 2
  },
  {
    content:
      "La facturation se fait automatiquement le 1er de chaque mois. Vous pouvez consulter vos factures dans Paramètres > Facturation.",
    metadata: { category: "billing", lang: "fr" },
  },
  {
    content:
      "Pour ajouter un nouveau membre à votre équipe, allez dans Paramètres > Équipe > Inviter un membre. Saisissez son email et choisissez son rôle.",
    metadata: { category: "team", lang: "fr" },
  },
  {
    content:
      "Notre API REST est accessible à https://api.example.com/v2. L'authentification se fait par Bearer token dans le header Authorization.",
    metadata: { category: "api", lang: "fr" },
  },
  {
    content:
      "En cas de panne, consultez notre page de statut à https://status.example.com. Vous pouvez aussi contacter le support à support@example.com.",
    metadata: { category: "support", lang: "fr" },
  },
  {
    content:
      "Les exports de données sont disponibles au format CSV et JSON. Allez dans Données > Exporter et sélectionnez le format souhaité.",
    metadata: { category: "data", lang: "fr" },
  },
  {
    content:
      "L'intégration Slack est disponible dans Paramètres > Intégrations > Slack. Cliquez sur Connecter et autorisez l'accès.",
    metadata: { category: "integrations", lang: "fr" },
  },
  {
    content:
      "Pour activer l'authentification à deux facteurs (2FA), allez dans Paramètres > Sécurité > 2FA. Scannez le QR code avec votre app d'authentification.",
    metadata: { category: "auth", lang: "fr" },
  },
  {
    content:
      "Les webhooks permettent de recevoir des notifications en temps réel. Configurez-les dans Paramètres > API > Webhooks.",
    metadata: { category: "api", lang: "fr" },
  },
];

async function ingest() {
  // --- Connexion a PostgreSQL ---
  const client = new Client(DB_CONFIG);             // Creer le client avec la config
  await client.connect();                           // Se connecter

  // --- Instance d'embeddings ---
  // OllamaEmbeddings appelle Ollama pour transformer du texte en vecteur
  const embeddings = new OllamaEmbeddings({
    model: EMBEDDING_MODEL,                         // nomic-embed-text
    baseUrl: OLLAMA_BASE_URL,                       // http://localhost:11434
  });

  console.log(`Ingesting ${SAMPLE_DOCS.length} documents...`);

  // --- Boucle d'ingestion ---
  // Pour chaque document :
  //   1. Generer son embedding (vecteur de 768 nombres)
  //   2. L'inserer dans PostgreSQL avec son contenu et metadata
  for (const doc of SAMPLE_DOCS) {
    // nomic-embed-text necessite le prefix "search_document: " pour les documents
    // et "search_query: " pour les requetes — sinon la similarite est tres faible
    const [vector] = await embeddings.embedDocuments([`search_document: ${doc.content}`]);

    // INSERT dans PostgreSQL
    // $1 = contenu texte
    // $2 = metadata en JSON string
    // $3 = vecteur en JSON string (pgvector accepte le format "[0.1, 0.2, ...]")
    await client.query(
      "INSERT INTO documents (content, metadata, embedding) VALUES ($1, $2, $3)",
      [doc.content, JSON.stringify(doc.metadata), JSON.stringify(vector)]
    );

    console.log(`  [+] ${doc.content.slice(0, 60)}...`); // Afficher les 60 premiers caracteres
  }

  // --- Fermer la connexion ---
  await client.end();
  console.log(`\nDone! ${SAMPLE_DOCS.length} documents ingested.`);
}

// Lancer l'ingestion et gerer les erreurs
ingest().catch((err) => {
  console.error("Ingestion failed:", err.message);
  process.exit(1);
});
