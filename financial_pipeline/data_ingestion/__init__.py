"""
Data Ingestion Module.

This package handles fetching raw financial data from external sources.
Currently, it primarily uses `yfinance` to download historical market data.
"""
from .data_fetcher import fetch_financial_data

__all__ = ['fetch_financial_data']
