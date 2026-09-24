"""Backward-compatible imports for the relocated chat_agent capability.

New code should import from ``resilience_app.agents.chat_agent``.
"""
from resilience_app.agents.chat_agent.page import page_chat

__all__ = ['page_chat']
