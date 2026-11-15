import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from config import BASE_DIR, DEFAULT_CURRENCY_SYMBOL
from llm_engine.hf_manager import LocalHFEngine


class ToolCallPlan(BaseModel):
    tickers_to_analyze: List[str] = Field(
        default_factory=lambda: ["NIFTYBEES.NS", "JUNIORBEES.NS", "GOLDBEES.NS", "LIQUIDBEES.NS"],
        description="3 to 6 ETF or stock ticker symbols suited to the user's currency/market and risk profile",
    )
    news_search_query: str = Field(
        default="Nifty 50 RBI interest rates India ETF market outlook",
        description="Focused financial news search query for macro and market context",
    )
    risk_profile_normalized: str = Field(
        default="Moderate",
        description="Normalized risk profile: 'Conservative', 'Moderate', or 'Aggressive'",
    )
    rationale: str = Field(
        description="Brief 1-sentence reason for selecting these tickers and news query"
    )


class MCPInvestmentOrchestrator:
    """
    Connects to `mcp_investment/market_server.py` over `stdio`, inspects registered
    MCP tools, uses `LocalHFEngine.generate_structured()` (`ToolCallPlan`) to route
    tool arguments, executes all MCP tools, and synthesizes a personalized investment plan.
    """

    def __init__(self) -> None:
        self.hf_engine = LocalHFEngine()
        self.server_script: Path = BASE_DIR / "mcp_investment" / "market_server.py"

    async def _execute_mcp_pipeline_async(
        self,
        monthly_income: float,
        predicted_expenses: float,
        risk_profile: str,
        user_goal: str,
        currency_symbol: str = DEFAULT_CURRENCY_SYMBOL,
    ) -> Dict[str, Any]:
        monthly_surplus = max(0.0, round(monthly_income - predicted_expenses, 2))
        is_inr = currency_symbol.strip() == "₹"

        default_tickers = (
            ["NIFTYBEES.NS", "JUNIORBEES.NS", "GOLDBEES.NS", "LIQUIDBEES.NS", "BANKBEES.NS"]
            if is_inr
            else ["VTI", "VXUS", "BND", "SGOV", "GLD"]
        )
        default_news_q = (
            "Nifty 50 Sensex RBI SIP ETF market outlook"
            if is_inr
            else "S&P 500 ETF bond yields market outlook"
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        python_exe = sys.executable
        if "streamlit" in python_exe.lower():
            python_exe = "python"

        diagnostic = subprocess.run(
            [python_exe, "-c", f"import sys; sys.path.insert(0, r'{BASE_DIR}'); import mcp_investment.market_server"],
            capture_output=True,
            text=True,
            env=env,
        )
        if diagnostic.returncode != 0:
            raise RuntimeError(
                f"MCP Server failed pre-flight import check.\n\n"
                f"**Error Traceback:**\n{diagnostic.stderr}\n\n"
                f"**Stdout:**\n{diagnostic.stdout}"
            )

        server_params = StdioServerParameters(
            command=python_exe,
            args=[str(self.server_script)],
            env=env,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                available_tools = [
                    {"name": t.name, "description": t.description}
                    for t in tools_response.tools
                ]

                market_hint = (
                    "Use Indian NSE Yahoo Finance tickers ending with '.NS' (e.g., NIFTYBEES.NS, JUNIORBEES.NS, GOLDBEES.NS, LIQUIDBEES.NS, BANKBEES.NS, ITBEES.NS)."
                    if is_inr
                    else "Use US ETF tickers (e.g., VTI, VXUS, BND, SGOV, QQQM, GLD)."
                )
                router_system_prompt = (
                    "You are a local financial tool router. Given the available MCP tools and the user's "
                    "investable surplus, risk tolerance, and investment goal, output a structured ToolCallPlan "
                    f"specifying 4-5 diversified ETF/stock tickers to screen and a relevant market news query. {market_hint}"
                )
                router_user_prompt = (
                    f"Available MCP Tools: {json.dumps(available_tools)}\n"
                    f"Currency: {currency_symbol}\n"
                    f"Monthly Investable Surplus: {currency_symbol}{monthly_surplus:,.2f}\n"
                    f"Predicted Next Month Expenses: {currency_symbol}{predicted_expenses:,.2f}\n"
                    f"User Risk Profile: {risk_profile}\n"
                    f"User Goal / Preferences: {user_goal}"
                )

                try:
                    plan: ToolCallPlan = self.hf_engine.generate_structured(
                        system_prompt=router_system_prompt,
                        user_prompt=router_user_prompt,
                        schema_model=ToolCallPlan,
                        temperature=0.1,
                        max_tokens=400,
                    )
                    if not plan.tickers_to_analyze:
                        plan.tickers_to_analyze = default_tickers
                except Exception:
                    plan = ToolCallPlan(
                        tickers_to_analyze=default_tickers,
                        news_search_query=default_news_q,
                        risk_profile_normalized=risk_profile,
                        rationale="Default diversified core portfolio tickers.",
                    )

                alloc_res = await session.call_tool(
                    "calculate_investment_allocation",
                    arguments={
                        "monthly_surplus": monthly_surplus,
                        "risk_profile": plan.risk_profile_normalized,
                        "predicted_expenses_next_month": predicted_expenses,
                        "currency_symbol": currency_symbol,
                    },
                )
                metrics_res = await session.call_tool(
                    "get_asset_metrics",
                    arguments={"tickers": plan.tickers_to_analyze},
                )
                news_res = await session.call_tool(
                    "fetch_market_news",
                    arguments={"query": plan.news_search_query, "max_items": 5},
                )

                allocation_data = self._extract_mcp_json(alloc_res)
                metrics_data = self._extract_mcp_json(metrics_res)
                news_data = self._extract_mcp_json(news_res)

        cs = currency_symbol
        advisor_system_prompt = (
            "You are LocalFinAI's Fiduciary Investment Advisor running 100% locally. "
            f"All monetary amounts are in '{cs}'. Always format currency values with '{cs}'. "
            "Synthesize the deterministic SIP allocation, live yfinance ETF/asset metrics, and RSS market headlines "
            "into a clear, structured investment recommendation. Explicitly reference the user's RL-predicted "
            f"next-month expenses, emergency fund target, exact monthly {cs} SIP per asset class, and 1Y/3Y CAGR "
            "and max drawdown figures from the tool output."
        )
        advisor_user_prompt = (
            f"User Financial Profile (Currency: {cs}):\n"
            f"- Monthly Income: {cs}{monthly_income:,.2f}\n"
            f"- RL Predicted Next-Month Expenses: {cs}{predicted_expenses:,.2f}\n"
            f"- Investable Monthly Surplus: {cs}{monthly_surplus:,.2f}\n"
            f"- Risk Profile: {plan.risk_profile_normalized}\n"
            f"- Goal: {user_goal}\n\n"
            f"MCP Tool 1 (calculate_investment_allocation):\n{json.dumps(allocation_data, indent=2)}\n\n"
            f"MCP Tool 2 (get_asset_metrics via yfinance):\n{json.dumps(metrics_data, indent=2)}\n\n"
            f"MCP Tool 3 (fetch_market_news via RSS):\n{json.dumps(news_data, indent=2)}"
        )

        recommendation_memo = self.hf_engine.generate_text(
            system_prompt=advisor_system_prompt,
            user_prompt=advisor_user_prompt,
            temperature=0.2,
            max_tokens=850,
        )

        return {
            "available_mcp_tools": available_tools,
            "tool_call_plan": plan.model_dump(),
            "allocation_output": allocation_data,
            "asset_metrics_output": metrics_data,
            "market_news_output": news_data,
            "advisor_recommendation": recommendation_memo,
        }

    @staticmethod
    def _extract_mcp_json(tool_result: Any) -> Any:
        if tool_result is None:
            return {}

        structured = getattr(tool_result, "structured_content", None) or getattr(
            tool_result, "structuredContent", None
        )
        if structured is not None:
            if isinstance(structured, dict) and list(structured.keys()) == ["result"]:
                return structured["result"]
            return structured

        content_blocks = getattr(tool_result, "content", None) or []
        texts = [
            getattr(block, "text", "").strip()
            for block in content_blocks
            if hasattr(block, "text") and getattr(block, "text", "").strip()
        ]
        if not texts:
            return {}
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except Exception:
                return texts[0]

        parsed_items = []
        for t in texts:
            try:
                parsed_items.append(json.loads(t))
            except Exception:
                parsed_items.append(t)
        return parsed_items

    def run_advisor(
        self,
        monthly_income: float,
        predicted_expenses: float,
        risk_profile: str,
        user_goal: str,
        currency_symbol: str = DEFAULT_CURRENCY_SYMBOL,
    ) -> Dict[str, Any]:
        return asyncio.run(
            self._execute_mcp_pipeline_async(
                monthly_income=monthly_income,
                predicted_expenses=predicted_expenses,
                risk_profile=risk_profile,
                user_goal=user_goal,
                currency_symbol=currency_symbol,
            )
        )