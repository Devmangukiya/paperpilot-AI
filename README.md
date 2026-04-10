# 📄 Daily Research Paper Agent

An AI-powered agent that **automatically fetches, ranks, and summarises one highly relevant research paper every day** based on your chosen keywords.

Built with Python, Groq LLM, arXiv, and sentence-transformers — 100% free and open-source.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔍 **Paper Retrieval** | Fetches papers from arXiv using boolean keyword combinations |
| 🗂️ **Smart Filtering** | Date window (last N days) + keyword/abbreviation matching |
| 🧠 **Hybrid Ranking** | Embedding similarity (sentence-transformers) + LLM relevance scoring (Groq) |
| 📝 **Structured Summary** | 3-5 line summary, relevance explanation, 3 key insights |
| 🔁 **Daily Automation** | Scheduler runs once per day at a configurable time |
| 💾 **Caching** | Avoids re-delivering the same paper |
| 📡 **Optional REST API** | FastAPI server for HTTP access |
| 🖥️ **CLI Interface** | Full-featured command-line tool with `--keywords`, `--schedule`, `--json` flags |

---

## 🏗️ Architecture

```
project/
├── app/
│   ├── main.py               ← CLI entry point
│   ├── api.py                ← Optional FastAPI server
│   ├── config.py             ← All configuration (env-driven)
│   ├── models/
│   │   └── paper.py          ← Paper + AgentResult dataclasses
│   ├── services/
│   │   ├── retrieval.py      ← arXiv fetcher
│   │   ├── filter.py         ← Date + keyword filter
│   │   └── groq_client.py    ← Groq REST API wrapper
│   ├── agents/
│   │   ├── ranker.py         ← Hybrid ranking engine
│   │   └── summariser.py     ← LLM summarisation agent
│   ├── pipelines/
│   │   ├── daily_pipeline.py ← Full end-to-end orchestrator
│   │   └── scheduler.py      ← Daily job scheduler
│   └── utils/
│       ├── logger.py         ← Rotating file + console logger
│       └── cache.py          ← JSON-based deduplication cache
├── data/
│   ├── cache/                ← seen_papers.json
│   ├── logs/                 ← agent.log (rotating)
│   └── output/               ← paper_YYYY-MM-DD.json results
├── tests/                    ← pytest test suite
├── requirements.txt
├── .env.example
└── README.md
```

### Pipeline Flow

```
Keywords
   │
   ▼
[ArxivRetriever] ──► raw papers (up to MAX_RESULTS)
   │
   ▼
[PaperFilter] ──► date + keyword check
   │
   ▼
[PaperCache] ──► remove already-seen papers
   │
   ▼
[HybridRanker]
  ├── SentenceTransformer (local) ── embedding cosine similarity
  └── GroqClient (LLM) ────────── relevance score 1-10
   │
   ▼
[PaperSummariser] ──► summary, why_relevant, key_insights
   │
   ▼
JSON output file  +  Console display
```

---

## ⚡ Quick Start

### 1. Clone / Download

```bash
git clone https://github.com/yourname/daily-research-agent.git
cd daily-research-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` embedding model (~90 MB). This is cached locally after the first download.

### 4. Set Up API Key

```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

Get a **free** Groq API key at: https://console.groq.com

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 🚀 Usage

### Run Once (Default Keywords)

```bash
python -m app.main
```

### Run Once with Custom Keywords

```bash
python -m app.main --keywords "finance, AI, federated learning, XAI"
```

### Output as JSON (for piping / scripting)

```bash
python -m app.main --keywords "finance, AI" --json
```

### Start Daily Scheduler

```bash
python -m app.main --schedule --keywords "finance, AI, federated learning"
```

The agent runs immediately on start, then again every day at `DAILY_RUN_TIME` (default 08:00).

### Start FastAPI Server

```bash
uvicorn app.api:app --reload --port 8000
```

Then call:

```bash
# Run pipeline via HTTP POST
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"keywords": ["finance", "AI", "federated learning"]}'

# Get latest result
curl http://localhost:8000/latest
```

Interactive API docs: http://localhost:8000/docs

---

## 📊 Example Output

```
══════════════════════════════════════════════════════════════════════
  📄  DAILY RESEARCH PAPER AGENT
══════════════════════════════════════════════════════════════════════

🏆  TITLE : FedXAI-Finance: Federated Explainable AI for Credit Risk Assessment
             Across Decentralised Banking Institutions
👥  AUTHORS: Zhang Wei, Priya Sharma, João Costa … (+2 more)
📅  DATE  : 2024-11-14
🔗  LINK  : https://arxiv.org/abs/2411.09823
⭐  SCORE : 8.74 / 10

──────────────────────────────────────────────────────────────────────
📝  SUMMARY
   This paper proposes FedXAI-Finance, a novel framework that combines
   federated learning with explainable AI techniques to enable
   collaborative credit risk modelling across multiple banks without
   sharing raw customer data. The system uses SHAP values to provide
   locally interpretable predictions while a global federated model
   is trained using differential privacy guarantees. Experiments on
   real-world lending data from three partner institutions show a 12%
   improvement in AUC over single-institution baselines.

🎯  WHY RELEVANT
   Directly addresses the intersection of federated learning, explainable
   AI, and financial risk — matching all four of your tracked topics.

💡  KEY INSIGHTS
   • Privacy-preserving FL enables multi-bank model training without
     data sharing, addressing a key regulatory barrier in finance
   • SHAP-based local explanations satisfy GDPR Article 22 requirements
     for automated decision explanations
   • The differential privacy mechanism adds only 0.3% AUC overhead
     while providing ε=1.0 privacy guarantees

══════════════════════════════════════════════════════════════════════
```

**JSON output file** (`data/output/paper_2024-11-15.json`):

```json
{
  "title": "FedXAI-Finance: Federated Explainable AI for Credit Risk Assessment...",
  "authors": "Zhang Wei, Priya Sharma, João Costa … (+2 more)",
  "published": "2024-11-14",
  "summary": "This paper proposes FedXAI-Finance...",
  "why_relevant": "Directly addresses the intersection of federated learning...",
  "key_insights": [
    "Privacy-preserving FL enables multi-bank model training without data sharing",
    "SHAP-based local explanations satisfy GDPR Article 22 requirements",
    "The differential privacy mechanism adds only 0.3% AUC overhead"
  ],
  "link": "https://arxiv.org/abs/2411.09823",
  "score": 8.74
}
```

---

## ⚙️ Configuration Reference

All settings can be overridden in your `.env` file:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama3-70b-8192` | Groq model to use |
| `MAX_RESULTS` | `30` | Max papers to fetch from arXiv |
| `DATE_LOOKBACK_DAYS` | `365` | Only consider papers from last N days |
| `EMBEDDING_WEIGHT` | `0.4` | Weight of embedding score in hybrid ranking |
| `LLM_WEIGHT` | `0.6` | Weight of LLM score in hybrid ranking |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `DAILY_RUN_TIME` | `08:00` | Time to run the daily scheduled job |

---

## 🧪 Running Tests

```bash
pytest
```

Expected output:
```
tests/test_cache.py ......       [ 5 passed ]
tests/test_filter.py ......      [ 6 passed ]
tests/test_retrieval.py ....     [ 4 passed ]
tests/test_groq_client.py ....   [ 4 passed ]
tests/test_ranker.py ...         [ 3 passed ]
```

---

## 🔄 Cron Setup (Alternative to `--schedule`)

If you prefer system cron over the Python scheduler:

```bash
# Edit crontab
crontab -e

# Add this line to run at 8 AM every day
0 8 * * * cd /path/to/project && /path/to/venv/bin/python -m app.main --keywords "finance, AI, federated learning, XAI" >> /path/to/project/data/logs/cron.log 2>&1
```

---

## 🗺️ Future Improvements

- [ ] **Multi-source retrieval** — add Semantic Scholar, PubMed, SSRN
- [ ] **Email / Slack delivery** — send daily digest via SMTP or Webhook
- [ ] **Web UI dashboard** — React frontend showing history + trends
- [ ] **Citation graph analysis** — rank papers by downstream citation impact
- [ ] **Topic drift detection** — alert when a new sub-field emerges
- [ ] **PDF full-text ingestion** — summarise full paper, not just abstract
- [ ] **User feedback loop** — thumbs up/down to fine-tune ranking weights
- [ ] **Vector database** — replace JSON cache with ChromaDB for semantic deduplication

---

## 📦 Dependencies

| Library | Purpose | License |
|---|---|---|
| `arxiv` | arXiv API client | MIT |
| `sentence-transformers` | Local embedding model | Apache 2.0 |
| `requests` | HTTP calls to Groq | Apache 2.0 |
| `python-dotenv` | `.env` file loader | BSD |
| `schedule` | Python job scheduler | MIT |
| `fastapi` | Optional REST API | MIT |
| `numpy` | Vector maths | BSD |

---

## 📄 License

MIT — free to use, modify, and distribute.
