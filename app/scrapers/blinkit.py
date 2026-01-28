"""Blinkit scraper - currently limited due to anti-bot protection."""
from typing import List
from .base import BaseScraper, ProductResult


class BlinkitScraper(BaseScraper):
    """Scraper for Blinkit.
    
    Note: Blinkit uses aggressive Cloudflare protection which blocks
    automated scraping. This scraper returns empty results.
    For production use, consider using their official API if available,
    or a proxy/CAPTCHA solving service.
    """
    
    PLATFORM_NAME = "Blinkit"
    BASE_URL = "https://blinkit.com"
    
    def __init__(self, pincode: str = "560087"):
        super().__init__(pincode)
        
    async def search(self, query: str) -> List[ProductResult]:
        """Search for products on Blinkit.
        
        Currently returns empty results due to Cloudflare blocking.
        """
        # Blinkit blocks automated requests with Cloudflare
        # To enable this scraper, you would need:
        # 1. A proxy rotation service
        # 2. CAPTCHA solving service
        # 3. Or use their official API with proper authentication
        
        return []
