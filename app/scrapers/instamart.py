"""Swiggy Instamart scraper - currently limited due to anti-bot protection."""
from typing import List
from .base import BaseScraper, ProductResult


class InstamartScraper(BaseScraper):
    """Scraper for Swiggy Instamart.
    
    Note: Instamart requires location setup and uses heavy JavaScript
    rendering which makes scraping difficult. This scraper returns empty results.
    For production use, consider using their official API if available.
    """
    
    PLATFORM_NAME = "Instamart"
    BASE_URL = "https://www.swiggy.com/instamart"
    
    def __init__(self, pincode: str = "560087"):
        super().__init__(pincode)
        
    async def search(self, query: str) -> List[ProductResult]:
        """Search for products on Instamart.
        
        Currently returns empty results due to complex authentication
        and location requirements.
        """
        # Instamart requires:
        # 1. Location/address setup before any search
        # 2. Complex session management
        # 3. Heavy JavaScript rendering
        
        return []
