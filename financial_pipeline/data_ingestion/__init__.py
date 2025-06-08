# This file makes Python treat the `data_ingestion` directory as a package.
from .data_fetcher import fetch_financial_data

__all__ = ['fetch_financial_data']
