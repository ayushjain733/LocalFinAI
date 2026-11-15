from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# ============================================================================
# Directory & Storage Paths
# ============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent
MODELS_DIR: Path = BASE_DIR / "models"
LLM_DIR: Path = MODELS_DIR / "llm"
EMBEDDING_DIR: Path = MODELS_DIR / "embeddings"
RL_MODEL_DIR: Path = MODELS_DIR / "rl_agent"
DB_DIR: Path = BASE_DIR / "data"

for directory in (MODELS_DIR, LLM_DIR, EMBEDDING_DIR, RL_MODEL_DIR, DB_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DB_PATH: Path = DB_DIR / "localfinai.db"
DATABASE_URL: str = f"sqlite:///{DB_PATH}"
RL_MODEL_PATH: Path = RL_MODEL_DIR / "ppo_expense_forecaster.zip"

# ============================================================================
# Currency & Regional Configuration (Defaults to Indian Rupee ₹)
# ============================================================================
SUPPORTED_CURRENCIES: Dict[str, str] = {
    "INR (₹)": "₹",
    "USD ($)": "$",
    "EUR (€)": "€",
    "GBP (£)": "£",
}
DEFAULT_CURRENCY_LABEL: str = "INR (₹)"
DEFAULT_CURRENCY_SYMBOL: str = "₹"

# ============================================================================
# Offline Account Recovery Security Questions
# ============================================================================
SECURITY_QUESTIONS: List[str] = [
    "What was the name of your first school?",
    "In what city or town was your first job?",
    "What was the name of your childhood best friend?",
    "What was the make or model of your first vehicle?",
    "What is your maternal grandmother's first name?",
    "What was the name of your first pet?",
]

# ============================================================================
# Hugging Face Hub & Hardware Ceiling Configuration (16GB RAM CPU-Only)
# ============================================================================
HF_LLM_REPO_ID: str = "Qwen/Qwen2.5-3B-Instruct-GGUF"
HF_LLM_FILENAME: str = "qwen2.5-3b-instruct-q4_k_m.gguf"

HF_EMBEDDING_REPO_ID: str = "sentence-transformers/all-MiniLM-L6-v2"

LLM_N_CTX: int = 4096
LLM_N_BATCH: int = 256
LLM_USE_MMAP: bool = True
LLM_N_THREADS: int = max(1, (os.cpu_count() or 4) - 2)

# Ingestion & Categorization Thresholds
EXTRACTION_BATCH_SIZE: int = 15
MIN_PAGE_TEXT_CHARS: int = 30
SIMILARITY_THRESHOLD: float = 0.78

# ============================================================================
# Financial Category Taxonomy & Domain Constants
# ============================================================================
EXPENSE_CATEGORIES: List[str] = [
    "Housing",
    "Utilities",
    "Groceries",
    "Dining & Cafe",
    "Transport & Fuel",
    "Shopping & Retail",
    "Healthcare",
    "Subscriptions & Media",
    "Entertainment & Leisure",
    "Travel",
    "Miscellaneous",
]

INCOME_CATEGORIES: List[str] = [
    "Salary & Payroll",
    "Investments & Dividends",
    "Refunds & Transfers",
]

ALL_CATEGORIES: List[str] = EXPENSE_CATEGORIES + INCOME_CATEGORIES

# Essential categories incur a 1.5x under-prediction penalty in the RL environment
ESSENTIAL_CATEGORIES: set[str] = {"Housing", "Utilities", "Groceries"}

# Deterministic Tier-1 Regex Patterns (Indian UPI/Merchants + Global Merchants)
DEFAULT_REGEX_RULES: Dict[str, str] = {
    r"\b(rent|mortgage|hoa|property\s*tax|apartments|nobroker|mygate|society\s*maint)\b": "Housing",
    r"\b(electric|water\s*bill|bescom|bwssb|mahadiscom|tata\s*power|adani\s*elec|jio\s*fiber|airtel|vodafone|vi\s*prepaid|bsnl|act\s*fibernet|gas\s*agency|indane|bharatgas|hp\s*gas|pg&e|coned|comcast|verizon)\b": "Utilities",
    r"\b(blinkit|zepto|bigbasket|instamart|dmart|reliance\s*smart|more\s*retail|nature\s*basket|ratnadeep|nilgiris|whole\s*foods|trader\s*joe|kroger|safeway|aldi|costco|grocery|supermarket)\b": "Groceries",
    r"\b(swiggy|zomato|eatfit|starbucks|third\s*wave|blue\s*tokai|chaayos|haldiram|mcdonald|dominos|pizza\s*hut|kfc|burger\s*king|barbeque\s*nation|chipotle|doordash|ubereats|cafe|coffee|bistro|restaurant|bakery|sweet)\b": "Dining & Cafe",
    r"\b(uber|ola|rapido|namma\s*yatri|blu\s*smart|indian\s*oil|iocl|hpcl|bpcl|bharat\s*petroleum|hindustan\s*petro|shell|fastag|nhai|bmrc|metro|irctc|petrol|fuel|parking|toll|lyft|chevron)\b": "Transport & Fuel",
    r"\b(amazon|amzn|flipkart|myntra|ajio|nykaa|meesho|croma|reliance\s*digital|vijay\s*sales|decathlon|titan|tanishq|ikea|zara|h&m|westside|pantaloons|lifestyle|best\s*buy|apple\s*store|target)\b": "Shopping & Retail",
    r"\b(apollo|pharmeasy|1mg|tata\s*1mg|medplus|netmeds|practo|manipal\s*hosp|fortis|max\s*health|narayana\s*health|dr\s*lal|metropolis|diagnostic|pharmacy|chemist|dental|optometry|hospital|clinic|medical|cvs|walgreens)\b": "Healthcare",
    r"\b(netflix|spotify|hotstar|jiocinema|sonyliv|zee5|prime\s*video|youtube\s*prem|apple\.com/bill|icloud|chatgpt|openai|audible|cult\.fit|cultfit|gym|hulu|disney\+)\b": "Subscriptions & Media",
    r"\b(bookmyshow|pvr|inox|cinepolis|wonderla|steam\s*games|playstation|nintendo|amc\s*theatres|ticketmaster|bowling|museum)\b": "Entertainment & Leisure",
    r"\b(makemytrip|goibibo|cleartrip|yatra|indigo|air\s*india|vistara|akasa|spicejet|redbus|abhibus|oyo|airbnb|marriott|hilton|taj\s*hotels|ihcl|expedia|booking\.com|delta\s*air|united\s*air)\b": "Travel",
    r"\b(payroll|salary|sal\s*credit|neft\s*cr|ach\s*credit|direct\s*dep|stipend|gusto|adp)\b": "Salary & Payroll",
    r"\b(zerodha|groww|upstox|kuvera|coin|cams|kfintech|mutual\s*fund|ach\s*dr\s*mf|nse\s*clea|indian\s*clearing|dividend|interest\s*paid|int\.pd|vanguard|fidelity|schwab)\b": "Investments & Dividends",
}

# Monthly Seasonality Multipliers (Months 1..12) per Expense Category
SEASONALITY_INDEX: Dict[int, Dict[str, float]] = {
    m: {
        "Housing": 1.00,
        "Utilities": 1.18 if m in (4, 5, 6) else 0.94,
        "Groceries": 1.12 if m in (10, 11, 12) else 0.98,
        "Dining & Cafe": 1.15 if m in (5, 10, 11, 12) else 0.95,
        "Transport & Fuel": 1.10 if m in (4, 5, 10, 11, 12) else 0.96,
        "Shopping & Retail": 1.35 if m in (10, 11, 12) else 0.90,
        "Healthcare": 1.15 if m in (1, 7, 8) else 0.97,
        "Subscriptions & Media": 1.00,
        "Entertainment & Leisure": 1.20 if m in (5, 10, 11, 12) else 0.93,
        "Travel": 1.40 if m in (4, 5, 10, 11, 12) else 0.85,
        "Miscellaneous": 1.08 if m in (10, 11, 12) else 0.98,
    }
    for m in range(1, 13)
}