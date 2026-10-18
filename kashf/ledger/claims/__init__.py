"""Registered claims. Importing this package registers every claim with the ledger."""

from kashf.ledger.claims import amounts, calendar, channels, quality

__all__ = ["amounts", "calendar", "channels", "quality"]
