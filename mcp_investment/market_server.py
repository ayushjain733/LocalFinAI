import math
from typing import Any, Dict, List
from urllib.parse import quote_plus

import feedparser
import yfinance as yf

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

mcp = MCPServer("LocalFinAI-Market-Server")


@mcp.tool()
def get_asset_metrics(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetches public market data via `yfinance` for a list of ticker symbols
    (supports Indian NSE `.NS` / BSE `.BO` tickers like `NIFTYBEES.NS`, `GOLDBEES.NS`,
    as well as global ETFs like `VTI`, `QQQM`).
    """
    results: Dict[str, Dict[str, Any]] = {}
    clean_tickers = [t.strip().upper() for t in tickers if t and t.strip()][:8]

    for symbol in clean_tickers:
        try:
            ticker_obj = yf.Ticker(symbol)
            hist = ticker_obj.history(period="3y", interval="1d", auto_adjust=True)
            if hist.empty or "Close" not in hist:
                results[symbol] = {"error": "No historical price series returned"}
                continue

            closes = hist["Close"].dropna()
            if closes.empty:
                results[symbol] = {"error": "Empty closing price series"}
                continue

            current_price = float(closes.iloc[-1])

            if len(closes) >= 252:
                price_1y_ago = float(closes.iloc[-252])
                cagr_1y = ((current_price / price_1y_ago) - 1.0) * 100.0
            else:
                cagr_1y = ((current_price / float(closes.iloc[0])) - 1.0) * 100.0

            years_span = max(len(closes) / 252.0, 0.25)
            cagr_3y = (((current_price / float(closes.iloc[0])) ** (1.0 / years_span)) - 1.0) * 100.0

            daily_returns = closes.pct_change().dropna()
            volatility_ann = (
                float(daily_returns.std() * math.sqrt(252) * 100.0)
                if not daily_returns.empty
                else 0.0
            )

            rolling_peak = closes.cummax()
            drawdowns = (closes - rolling_peak) / rolling_peak
            max_drawdown = float(drawdowns.min() * 100.0)

            info: Dict[str, Any] = {}
            try:
                info = ticker_obj.info or {}
            except Exception:
                info = {}

            pe_ratio = info.get("trailingPE") or info.get("forwardPE")
            raw_div_yield = (
                info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0.0
            )
            div_yield_pct = (
                float(raw_div_yield * 100.0)
                if raw_div_yield and raw_div_yield < 1.0
                else float(raw_div_yield or 0.0)
            )

            results[symbol] = {
                "price": round(current_price, 2),
                "currency": info.get("currency", "INR" if symbol.endswith((".NS", ".BO")) else "USD"),
                "cagr_1y_pct": round(cagr_1y, 2),
                "cagr_3y_pct": round(cagr_3y, 2),
                "annualized_volatility_pct": round(volatility_ann, 2),
                "max_drawdown_pct": round(max_drawdown, 2),
                "pe_ratio": round(float(pe_ratio), 2) if pe_ratio else None,
                "dividend_yield_pct": round(div_yield_pct, 2),
            }
        except Exception as exc:
            results[symbol] = {"error": str(exc)}

    return results


@mcp.tool()
def fetch_market_news(query: str, max_items: int = 5) -> List[Dict[str, str]]:
    """
    Fetches latest public financial headlines from Indian & global RSS feeds using `feedparser`.
    """
    encoded_q = quote_plus(query.strip() or "Nifty 50 RBI ETF market")
    rss_urls = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={encoded_q}&region=IN&lang=en-IN",
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    ]

    headlines: List[Dict[str, str]] = []
    seen_titles: set[str] = set()

    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = str(getattr(entry, "title", "")).strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                headlines.append(
                    {
                        "title": title,
                        "published": str(getattr(entry, "published", "Recent")),
                        "source": str(getattr(feed.feed, "title", "Financial RSS")),
                        "link": str(getattr(entry, "link", "")),
                    }
                )
                if len(headlines) >= max_items:
                    return headlines
        except Exception:
            continue

    return headlines[:max_items]


@mcp.tool()
def calculate_investment_allocation(
    monthly_surplus: float,
    risk_profile: str,
    predicted_expenses_next_month: float,
    currency_symbol: str = "₹",
) -> Dict[str, Any]:
    """
    Deterministic capital allocation calculator computing emergency fund targets,
    liquidity buffers, and monthly Systematic Investment Plan (SIP) splits in ₹ or $.
    """
    surplus = max(0.0, float(monthly_surplus))
    pred_exp = max(500.0, float(predicted_expenses_next_month))
    profile_key = (risk_profile or "Moderate").strip().lower()
    is_inr = currency_symbol.strip() == "₹" or "inr" in currency_symbol.lower()

    emergency_months = 6.0 if "conserv" in profile_key else (4.5 if "aggress" in profile_key else 6.0)
    emergency_fund_target = round(pred_exp * emergency_months, 2)

    emergency_buffer_contribution = round(surplus * 0.20, 2)
    investable_sip_total = round(surplus - emergency_buffer_contribution, 2)

    if is_inr:
        if "conserv" in profile_key:
            weights = {
                "Liquid / Gilt Debt ETF & FD Buffer (LIQUIDBEES.NS / LTGILTBEES.NS)": 0.50,
                "Nifty 50 Core Index ETF (NIFTYBEES.NS)": 0.30,
                "Nifty Next 50 / Quality Dividend (JUNIORBEES.NS)": 0.10,
                "Sovereign Gold / Gold ETF (GOLDBEES.NS)": 0.10,
            }
            recommended_tickers = ["LIQUIDBEES.NS", "LTGILTBEES.NS", "NIFTYBEES.NS", "GOLDBEES.NS"]
        elif "aggress" in profile_key:
            weights = {
                "Nifty 50 Core Index ETF (NIFTYBEES.NS)": 0.45,
                "Nifty Next 50 & Midcap Growth (JUNIORBEES.NS / MID150BEES.NS)": 0.30,
                "Sectoral IT / Banking ETFs (ITBEES.NS / BANKBEES.NS)": 0.15,
                "Gold & Liquid Hedge (GOLDBEES.NS / LIQUIDBEES.NS)": 0.10,
            }
            recommended_tickers = ["NIFTYBEES.NS", "JUNIORBEES.NS", "MID150BEES.NS", "ITBEES.NS", "GOLDBEES.NS"]
        else:
            weights = {
                "Nifty 50 Core Index ETF (NIFTYBEES.NS)": 0.45,
                "Nifty Next 50 Growth ETF (JUNIORBEES.NS)": 0.15,
                "Liquid & Gilt Debt ETFs (LIQUIDBEES.NS / LTGILTBEES.NS)": 0.25,
                "Gold ETF Hedge (GOLDBEES.NS)": 0.15,
            }
            recommended_tickers = ["NIFTYBEES.NS", "JUNIORBEES.NS", "LIQUIDBEES.NS", "LTGILTBEES.NS", "GOLDBEES.NS"]
    else:
        if "conserv" in profile_key:
            weights = {
                "Short-Term Treasury / Debt ETFs (SGOV / BND)": 0.50,
                "Broad Index Equity ETFs (VTI / VXUS)": 0.30,
                "Dividend Quality Equity (SCHD)": 0.10,
                "Gold / Inflation Hedge (GLD)": 0.10,
            }
            recommended_tickers = ["SGOV", "BND", "VTI", "SCHD", "GLD"]
        elif "aggress" in profile_key:
            weights = {
                "Core S&P 500 & Total Market ETFs (VTI / VOO)": 0.50,
                "Growth & Nasdaq-100 ETFs (QQQM)": 0.25,
                "International Developed/EM Equity (VXUS)": 0.15,
                "Short-Duration Bond / Tactical Cash (BND)": 0.10,
            }
            recommended_tickers = ["VTI", "QQQM", "VXUS", "BND"]
        else:
            weights = {
                "Core Total Stock Market ETFs (VTI / VOO)": 0.45,
                "International Equity ETF (VXUS)": 0.15,
                "Investment-Grade Bond & Treasury ETFs (BND / SGOV)": 0.30,
                "Gold / Real Asset Diversifier (GLD)": 0.10,
            }
            recommended_tickers = ["VTI", "VXUS", "BND", "SGOV", "GLD"]

    sip_breakdown = {
        bucket: round(investable_sip_total * weight, 2)
        for bucket, weight in weights.items()
    }

    return {
        "currency": currency_symbol,
        "risk_profile": risk_profile.title(),
        "monthly_surplus": round(surplus, 2),
        "predicted_expenses_next_month": round(pred_exp, 2),
        "emergency_fund_6m_target": emergency_fund_target,
        "monthly_emergency_buffer_topup": emergency_buffer_contribution,
        "net_monthly_sip_deployable": investable_sip_total,
        "suggested_sip_allocation": sip_breakdown,
        "recommended_tickers_to_screen": recommended_tickers,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")