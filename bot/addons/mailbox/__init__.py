"""Mailbox addon — IMAP poller and task-card delivery via Telegram.

Self-contained: brings its own ``config`` / ``credentials`` /
``imap_client`` / ``parsers`` modules (ported from
``freelance-mailbox-mcp``) plus an aiogram-based ``handlers`` module and
a background ``poller``.
"""
