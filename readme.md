# 🛡️ LocalFinAI — 100% Local, Privacy-First Personal Finance & Investment Intelligence

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)
![LLM](https://img.shields.io/badge/Local%20LLM-Qwen2.5--3B--Instruct--GGUF-orange)
![Embeddings](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-yellow)
![RL](https://img.shields.io/badge/RL-Gymnasium%20%2B%20SB3%20PPO-brightgreen)
![MCP](https://img.shields.io/badge/Protocol-Model%20Context%20Protocol%20(MCP)-purple)
![Database](https://img.shields.io/badge/Database-SQLite%20WAL%20%2B%20SQLModel-lightgrey)

**LocalFinAI** is an end-to-end, locally hosted AI personal finance, expense forecasting, and investment intelligence platform built in Python.

Designed to run strictly **in-process on consumer CPUs within a 16 GB RAM ceiling**, LocalFinAI requires **zero external LLM APIs** (no OpenAI, no Anthropic, no Ollama server). Every bank statement, transaction row, merchant rule, and reinforcement learning policy stays on your local disk.

---

## ✨ Key Architectural Highlights

1. **100% Local In-Process AI Execution:**
   - **LLM Engine:** Runs `Qwen/Qwen2.5-3B-Instruct-GGUF` (`qwen2.5-3b-instruct-q4_k_m.gguf`, ~1.93 GB) directly in Python via `llama-cpp-python` with grammar-constrained JSON Schema enforcement.
   - **Semantic Embedding Engine:** Runs `sentence-transformers/all-MiniLM-L6-v2` (~90 MB RAM) on CPU for sub-millisecond semantic transaction categorization.
   - **Offline-First Model Provisioning:** Models download automatically from Hugging Face Hub into `./models/` on first launch and lock into `local_files_only=True` thereafter.
   - **Deterministic Memory Governance:** Explicit `unload_llm()` + garbage collection frees ~2.1 GB of RAM on demand or automatically before CPU-intensive Reinforcement Learning training.

2. **Strict Network Boundary:**
   - Personal transactions, raw bank statements, and merchant descriptions **never** leave your machine.
   - Outbound calls occur only inside the isolated **Model Context Protocol (`MCP`)** server over `stdio` to fetch public market metrics (`yfinance`) and public financial RSS headlines (`feedparser`) using only your aggregate monthly investable surplus.

3. **Multi-User Isolated Workspaces & Offline Account Recovery:**
   - Local account authentication secured with `PBKDF2-HMAC-SHA256` (120,000 iterations + per-user cryptographic salt).
   - **Blank-Slate Onboarding:** Every new user starts with a clean, isolated workspace across all tabs.
   - **Dual-Factor Offline Password Recovery:** Recover forgotten passwords 100% offline via either a generated **Master Recovery Key (`LFAI-XXXX-XXXX-XXXX`)** (downloadable as an Emergency Recovery Kit `.txt`) or a **PBKDF2-hashed Security Question & Answer**.
   - **One-Click Full Workspace Backup & Restore:** Export and restore your entire workspace (SQLite transactions, learned merchant rules, RL forecasts, and trained PPO neural network weights) as a single `.zip` archive with Merge (deduplicating) or Replace modes.
   - **Permanent Account & Data Wipe:** Password-verified account deletion removes all database records and unlinks per-user RL model checkpoints from disk.

4. **Indian Rupee (`₹` / INR) & Global Multi-Currency Support:**
   - Native support for **`INR (₹)`**, **`USD ($)`**, **`EUR (€)`**, and **`GBP (£)`**.
   - Parses Indian bank statement formats including `Rs.` / `INR` / `₹` prefixes, Indian lakh/crore comma formatting (`1,25,000.00`), `Dr` / `Cr` suffixes, UPI narrations (`UPI/SWIGGY/...`), and split `Withdrawal Amt.` / `Deposit Amt.` CSV columns.
   - Automatically routes `INR (₹)` portfolios to Indian NSE ETFs (`NIFTYBEES.NS`, `JUNIORBEES.NS`, `GOLDBEES.NS`, `LIQUIDBEES.NS`, `BANKBEES.NS`) and Indian financial RSS feeds.

---

## 🏗️ System Architecture & Memory Budget

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│                         LOCAL HOST MACHINE (100% OFFLINE DATA)                    │
│                                                                                   │
│  ┌───────────────────────┐    ┌────────────────────────┐    ┌──────────────────┐  │
│  │ Ingestion Pipeline    │    │ 3-Tier Categorizer     │    │ SQLite + CRUD UI │  │
│  │ • pdfplumber (Stage A)├───►│ 1. Corrections & Regex ├───►│ • Per-User Auth  │  │
│  │ • RapidOCR (Stage B)  │    │ 2. MiniLM Cosine ≥0.78 │    │ • Staging & Full │  │
│  │ • Qwen2.5 JSON Schema │    │ 3. Qwen2.5 JSON Schema │    │   CRUD DataEd.   │  │
│  └───────────────────────┘    └────────────────────────┘    └────────┬─────────┘  │
│                                                                      │            │
│  ┌───────────────────────────────────────────────────────────────────▼─────────┐  │
│  │ LocalHFEngine Singleton (Memory-Managed: ~2.1 GB Peak)                      │  │
│  │ • Qwen2.5-3B-Instruct-GGUF (Q4_K_M, mmap=True, n_ctx=4096)                  │  │
│  │ • all-MiniLM-L6-v2 Embeddings (~90 MB CPU) + Explicit unload_llm() + GC     │  │
│  └──────────────┬────────────────────────────────────────────────────┬─────────┘  │
│                 │ unload_llm() before PPO training                   │            │
│  ┌──────────────▼────────────────────┐              ┌────────────────▼─────────┐  │
│  │ RL Expense Predictor (Gym + SB3)  │              │ MCP Client Orchestrator  │  │
│  │ • 3M WMA + Seasonality Baseline   ├─────────────►│ • Surplus = Inc - RL_Exp │  │
│  │ • PPO MlpPolicy [64, 64]          │  Investable  │ • ToolCallPlan JSON      │  │
│  │ • 1.5x Essential Under-pred Pen.  │   Surplus    └────────────────┬─────────┘  │
└──┴───────────────────────────────────┴───────────────────────────────┼─────────┴──┘
                                                                       │ stdio (Aggregate Numbers Only)
                                                        ┌──────────────▼─────────────┐
                                                        │ Isolated MCP Server        │
                                                        │ • get_asset_metrics (yf)   │
                                                        │ • fetch_market_news (RSS)  │
                                                        │ • calculate_allocation     │
                                                        └────────────────────────────┘
```

| Subsystem | Runtime Engine | Peak RAM Footprint | Isolation & Persistence Strategy |
| :--- | :--- | :--- | :--- |
| **LLM Inference** | `llama-cpp-python` (`qwen2.5-3b-instruct-q4_k_m.gguf`) | ~2.05 GB | `use_mmap=True`, `n_ctx=4096`, `n_batch=256`, explicit `unload_llm()` |
| **Semantic Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | ~90 MB | CPU-only (`device="cpu"`), L2-normalized cosine similarity |
| **OCR Fallback** | `rapidocr` + `onnxruntime` + `pypdfium2` | ~140 MB | Triggered only when PDF page text density is below 30 characters |
| **RL Forecaster** | `gymnasium` + `stable-baselines3` (`PPO`) | ~180 MB | `MlpPolicy` (`net_arch=[64, 64]`), per-user checkpoint (`ppo_expense_user_<id>.zip`) |
| **Database & UI** | `SQLite` (WAL) + `SQLModel` + `Streamlit` | ~150 MB | Per-user foreign keys, PBKDF2-SHA256 auth, portable `.zip` backups |

---

## 🔍 Core Modules & Technical Specifications

### 1. Two-Stage Document Extraction & Grammar-Constrained Parsing (`ingestion/pdf_extractor.py`)
- **Stage A (Native Digital PDFs):** Extracts structured tables and text lines using `pdfplumber`, with built-in support for password-protected/encrypted bank PDFs.
- **Stage B (Scanned PDFs & Images):** Automatically falls back to `rapidocr` (ONNX Runtime on CPU) rendered via `pypdfium2` when page text density is near zero.
- **Grammar-Constrained LLM Extraction:** Batches unparsed statement lines (15 lines/batch) into `LocalHFEngine.generate_structured()` using `llama-cpp-python` JSON Schema mode (`BatchTransactionSchema`) so output always conforms to strict Pydantic types.

### 2. 3-Tier Hybrid Categorizer (`ingestion/categorizer.py`)
1. **Tier 1 — User Overrides & Deterministic Regex (0ms latency):** Checks the user's `category_corrections` table first, followed by compiled regex rules covering Indian UPI/merchants (Swiggy, Zomato, Blinkit, Zepto, DMart, BESCOM, Airtel, Jio, IRCTC, Zerodha) and global merchants.
2. **Tier 2 — Semantic Cosine Similarity (~5ms latency):** Encodes merchant descriptions with `all-MiniLM-L6-v2` and matches against the user's verified historical ledger when cosine similarity is `>= 0.78`.
3. **Tier 3 — Local GGUF LLM Fallback:** Invokes `Qwen2.5-3B-Instruct-GGUF` with JSON Schema enforcement (`SingleCategorySchema`) only when similarity falls below `0.78`.

### 3. Human-in-the-Loop (HITL) Staging & Full CRUD Ledger (`database/db.py`, `app.py`)
- **Staging Review:** Edit extracted rows, skip unwanted rows, or add manual entries before committing to SQLite.
- **Full CRUD on Committed Ledger:** Directly **Create** (add rows via `+`), **Read** (search by keyword, filter by category or debit/credit, export CSV), **Update** (edit any committed cell inline), and **Delete** (check `🗑️ Delete?` or press `Delete` on selected rows) at any time.
- **Continuous Learning:** Every manual category edit in either the Staging Table or the Committed Ledger automatically updates `category_corrections` for future imports.

### 4. Reinforcement Learning Expense Predictor (`rl_engine/env.py`, `rl_engine/trainer.py`)
- **Custom Gymnasium Environment (`PersonalExpenseForecastEnv`):**
  - **Statistical Baseline:** 3-month weighted moving average (`[0.2, 0.3, 0.5]`) scaled by a monthly category seasonality index.
  - **Observation Space (`Box`, 18-dim):** `[month_sin, month_cos, *category_one_hot (11), rolling_3m_mean, rolling_3m_std, fixed_recurring_ratio, last_month_error_ratio, income_to_spend_ratio]`.
  - **Action Space (`Box`, 1-dim):** Continuous adjustment multiplier in `[-0.5, +0.5]` scaling the statistical baseline forecast.
  - **Asymmetric Reward Function:**
    $$\text{reward} = -\frac{\vert{}\text{predicted} - \text{actual}\vert{}}{\max(\text{actual}, 1.0)} \times \begin{cases} 1.5 & \text{if essential category and predicted} < \text{actual} \\ 1.0 & \text{otherwise} \end{cases}$$
    Applies a **1.5x penalty** when under-predicting essential categories (*Housing, Utilities, Groceries*).
- **Cold-Start Bootstrapping & Per-User Fine-Tuning:** Pre-trains on synthetic seasonal spending transitions and fine-tunes incrementally on the user's verified ledger after automatically unloading the GGUF LLM to stay well below the 16 GB RAM limit.

### 5. Local MCP Investment Advisor (`mcp_investment/market_server.py`, `mcp_investment/advisor_client.py`)
- **Standalone MCP Server (`stdio` transport):**
  - `calculate_investment_allocation`: Computes 6-month emergency fund targets, monthly liquidity buffer top-ups, and risk-adjusted SIP allocations across equity, debt/treasury, and gold ETFs.
  - `get_asset_metrics`: Uses `yfinance` to compute latest price, 1Y & 3Y CAGR, annualized volatility, maximum drawdown, P/E ratio, and dividend yield for NSE (`.NS`) or US ETFs.
  - `fetch_market_news`: Parses public financial RSS feeds (Economic Times, Moneycontrol, Yahoo Finance, CNBC) via `feedparser`.
- **LLM Tool Router (`MCPInvestmentOrchestrator`):** Connects over `stdio_client`, inspects registered MCP tools, generates a schema-validated `ToolCallPlan`, executes the tools, and synthesizes a fiduciary investment memo with `Qwen2.5-3B-Instruct`.

---

## 📁 Project Structure

```text
local_fin_ai/
├── app.py                        # Streamlit UI (Auth, Recovery, Backup, CRUD Ledger, Analytics, RL, MCP)
├── config.py                     # Hardware ceiling settings, HF repo IDs, currencies, regex rules
├── requirements.txt              # Python 3.11–3.13 compatible dependencies
├── .gitignore                    # Excludes SQLite DBs, GGUF models, backups, and venvs
├── data/                         # Auto-created local SQLite database directory (git-ignored)
├── models/                       # Auto-created local cache for GGUF, MiniLM & PPO zips (git-ignored)
├── llm_engine/
│   └── hf_manager.py             # Thread-safe singleton for llama-cpp-python + SentenceTransformer
├── database/
│   ├── models.py                 # SQLModel schemas (User, Transaction, CategoryCorrection, MonthlyForecast)
│   └── db.py                     # Auth, offline recovery, backup/restore, and per-user CRUD operations
├── ingestion/
│   ├── pdf_extractor.py          # pdfplumber + RapidOCR fallback + Qwen2.5 batch JSON parser
│   └── categorizer.py            # 3-Tier hybrid categorizer (Regex/Overrides -> MiniLM -> Qwen2.5)
├── rl_engine/
│   ├── env.py                    # Custom Gymnasium PersonalExpenseForecastEnv
│   ├── trainer.py                # Stable-Baselines3 PPO trainer, cold-start bootstrap & inference
│   └── insights.py               # Deterministic Pandas/SQL spend analytics + LLM commentary
└── mcp_investment/
    ├── market_server.py          # MCPServer / FastMCP stdio server (yfinance + RSS + SIP calculator)
    └── advisor_client.py         # MCP client orchestrator & structured LLM ToolCallPlan router
```

---

## 🚀 Installation & Quickstart

### Prerequisites
- **Python:** 3.11, 3.12, or 3.13 (64-bit)
- **RAM:** 8 GB minimum (16 GB recommended)
- **Disk Space:** ~2.5 GB free for local Hugging Face model weights (`Qwen2.5-3B-Instruct-GGUF` + `all-MiniLM-L6-v2`)

### 1. Clone the Repository & Create a Virtual Environment

**Windows (PowerShell):**
```powershell
git clone https://github.com/ayushjain733/LocalFinAI.git
cd local_fin_ai
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
git clone https://github.com/ayushjain733/LocalFinAI.git
cd local_fin_ai
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the Application

```bash
streamlit run app.py
```

> **First-Run Model Download:** The first time you trigger an AI action (statement extraction, semantic categorization, or AI commentary), `LocalHFEngine` will automatically download `qwen2.5-3b-instruct-q4_k_m.gguf` (~1.93 GB) and `all-MiniLM-L6-v2` (~90 MB) into `./models/`. All subsequent runs load directly from local disk with `local_files_only=True`.

---

## 📖 Usage Guide

1. **Create Your Account & Save Your Recovery Kit:**
   - Open the **Create New Account** tab, choose your default currency (`INR (₹)`, `USD ($)`, `EUR (€)`, or `GBP (£)`), and configure your offline Security Question.
   - Upon signup, click **⬇️ Download Emergency Recovery Kit (.txt)** to save your `LFAI-XXXX-XXXX-XXXX` Master Recovery Key.
2. **Import Statements or Add Transactions (`📒 Ledger & Statement Import`):**
   - Upload a native PDF, scanned PDF, image, or CSV bank statement (enter the PDF password if encrypted) and click **Extract Statement**.
   - Review the extracted rows in the **Staging Table** and click **✅ Commit Rows to Ledger**.
   - Edit, add, filter, or delete committed transactions anytime in the **Verified Transactions Ledger** table and click **💾 Save Ledger Changes**.
3. **Analyze Cashflow (`📊 Spend Analytics & AI Insights`):**
   - View exact monthly income, expenses, net cashflow, savings rate, category bar charts, Month-over-Month deltas, and top merchants.
   - Click **✨ Generate AI Spend Analysis** for a local Qwen2.5-3B executive briefing.
4. **Forecast Next Month's Spend (`📈 RL Expense Forecaster`):**
   - Click **🔮 Predict Next-Month Expenses** to compare the 3-month weighted moving average baseline against the PPO agent's category-level predictions.
   - Click **🧠 Train / Fine-Tune PPO Policy** to unload the LLM and fine-tune your personal RL agent on your latest verified ledger.
5. **Generate an Investment Plan (`🧭 MCP Investment Advisor`):**
   - Review your auto-populated Monthly Net Income and RL-Predicted Next-Month Expenses, select your Risk Tolerance Profile, and click **🧭 Run MCP Tool Router & Generate Investment Plan**.
6. **Backup or Restore Your Workspace (`📦 Backup & Restore Workspace` in Sidebar):**
   - Export a complete `.zip` snapshot of your account at any time or restore it onto another machine.

---

## 🔒 Privacy & Security Guarantees

- **Zero Telemetry / Zero Cloud LLMs:** No financial narrations, account numbers, or ledger entries are ever sent to external AI providers.
- **Cryptographic Credential Storage:** Passwords, security answers, and recovery keys are hashed with `PBKDF2-HMAC-SHA256` (120,000 iterations) using unique 128-bit per-user salts and verified in constant time via `hmac.compare_digest`.
- **Subprocess Isolation:** The MCP market server runs in an isolated child process and receives only three scalar values (`monthly_surplus`, `predicted_expenses_next_month`, and `risk_profile`).