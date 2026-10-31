"""Hybrid Ledger context engine plugin for Hermes."""

from .engine import HybridLedgerEngine


def register(ctx):
    """Register the HybridLedgerEngine with Hermes."""
    ctx.register_context_engine(HybridLedgerEngine())


__all__ = ["HybridLedgerEngine", "register"]
