<div align="center">
  <img src="assets/logo.svg" alt="NEXUS·ELASTIC" width="520"/>
  <br/><br/>

  [![License: MIT](https://img.shields.io/badge/License-MIT-F97316?style=flat-square)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.11+-F97316?style=flat-square&logo=python&logoColor=black)](#)
  [![Elasticsearch](https://img.shields.io/badge/Elasticsearch-vector_search-FBBF24?style=flat-square&logo=elasticsearch)](#)
  [![Gemini](https://img.shields.io/badge/Gemini-AI_Agent-4285F4?style=flat-square&logo=google)](#)
  [![Hackathon](https://img.shields.io/badge/Agent_Builder-Hackathon_2026-F97316?style=flat-square)](#)

  <br/>
  <p><strong>Agent d'intelligence financière autonome · Elasticsearch vector search · Gemini · Agent Builder Hackathon 2026</strong></p>
  <p><em>Analyse financière IA propulsée par Elasticsearch — search vectoriel, RAG financier, agent autonome</em></p>
</div>

---

## Présentation

**NEXUS·ELASTIC** est un agent d'intelligence financière construit pour l'**Agent Builder Hackathon 2026**. Il combine Elasticsearch (vector search + full-text) avec Gemini pour créer un assistant financier capable d'analyser des marchés, rechercher des patterns historiques, et générer des recommandations basées sur des données vectorisées.

---

## Architecture

```
NEXUS·ELASTIC — Agent financier
─────────────────────────────────────────────
  Market Data Feed
       │
       ▼
  Elasticsearch Index       Vector + full-text
  (données financières)     Embeddings Gemini
       │
       ▼
  RAG Pipeline              Retrieval-Augmented
  retrieve_context()        Generation
       │
       ▼
  Gemini Agent              Analyse + décision
  financial_analyst()       Recommandations
       │
       ▼
  Output: Report + Signals
```

---

## Stack

| Composant | Technologie |
|-----------|-------------|
| **Search** | Elasticsearch 8.x |
| **Embeddings** | Gemini text-embedding |
| **LLM** | Gemini Pro / Flash |
| **Framework** | Agent Builder (Google Cloud) |
| **API** | FastAPI + WebSocket |
| **DB** | Elasticsearch + SQLite |

---

## Installation

```bash
git clone https://github.com/Turbo31150/TradeOracle-Nexus-Elastic.git
cd TradeOracle-Nexus-Elastic
pip install -r requirements.txt
cp .env.example .env
# GOOGLE_API_KEY=... · ELASTICSEARCH_URL=...
python main.py
```

---

<div align="center">

**Franc Delmas (Turbo31150)** · Agent Builder Hackathon 2026 · MIT License

</div>
