"""Sentinel - DEMO-account supervisor for gold_monitor signals.

Consumes the deterministic signals gold_monitor persists in signals.db,
asks Claude to approve/veto each one and to manage the resulting demo
positions, and enforces hard risk limits in code no matter what the model
says. Every decision is logged (decisions.db) so the model's added value can
be measured against the plain deterministic path.
"""
