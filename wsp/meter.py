"""Budgets: metered forward passes and a context-token ledger.

The ledger is an *estimate* of how many tokens of wsp output the agent has pulled into its
context during the current session. It lets the agent stop (and write a handoff) before its
context window fills, and decides whether a full weight dump fits or the sandbox-inspection
fallback must be used.
"""
from __future__ import annotations

import math
import os
import time

DEFAULT_CONTEXT_TOKENS = 1_000_000   # Claude Opus 5.5 context window
CONTEXT_RESERVE = 0.25               # fraction kept free for the agent's own reasoning/tool calls
DEFAULT_FORWARD_BUDGET = 50          # largest forward-pass budget used in the blog post


class BudgetExhausted(RuntimeError):
    pass


def estimate_tokens(text: str, numeric: bool = False) -> int:
    # Numbers tokenize densely (~2.5 chars/token); prose ~3.5 chars/token. Conservative.
    return int(math.ceil(len(text) / (2.5 if numeric else 3.5)))


def context_window() -> int:
    return int(os.environ.get("WSP_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS))


def new_session(prev: dict | None, forward_budget: int | None, context_tokens: int | None) -> dict:
    window = int(context_tokens or context_window())
    prev = prev or {}
    return {
        "index": int(prev.get("index", 0)) + 1,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "forward_budget": int(forward_budget if forward_budget is not None
                              else prev.get("forward_budget", DEFAULT_FORWARD_BUDGET)),
        "forward_used": 0,
        "context_window": window,
        "context_limit": int(window * (1 - CONTEXT_RESERVE)),
        "context_used": 0,
    }


def charge_tokens(session: dict, n: int) -> None:
    session["context_used"] = int(session.get("context_used", 0)) + int(n)


def context_remaining(session: dict) -> int:
    return int(session["context_limit"]) - int(session.get("context_used", 0))


def forward_remaining(session: dict) -> int:
    return int(session["forward_budget"]) - int(session.get("forward_used", 0))


def charge_forward(session: dict, n: int = 1) -> None:
    if forward_remaining(session) < n:
        raise BudgetExhausted(
            f"forward-pass budget exhausted ({session['forward_used']}/{session['forward_budget']} used). "
            "Stop editing: pick the best version, write handoff.md, and finish (or ask the user for more budget)."
        )
    session["forward_used"] = int(session.get("forward_used", 0)) + n
