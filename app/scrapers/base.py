"""Base scraper class with robust scraping support."""
import asyncio
import random
from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass
from contextlib import asynccontextmanager
from fake_useragent import UserAgent
import httpx


@dataclass
class ProductResult:
    """Represents a product search result."""
    name: str
    price: float
    original_price: Optional[float]
    discount: Optional[str]
    platform: str
    url: str
    image_url: Optional[str]
    rating: Optional[float]
    available: bool = True
    delivery_time: Optional[str] = None


class BaseScraper(ABC):
    """Base class for all platform scrapers."""
    
    PLATFORM_NAME: str = "Base"
    BASE_URL: str = ""
    USE_BROWSER: bool = False  # Disabled by default - use HTTP first
    
    def __init__(self, pincode: str = "560087"):
        self.pincode = pincode
        self.ua = UserAgent()
        self.timeout = 30.0
        self._browser_available = None
        
    def get_headers(self) -> dict:
        """Get randomized headers to avoid detection."""
        return {
            "User-Agent": self.ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7,hi;q=0.6",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
        }
    
    async def delay(self, min_sec: float = 0.5, max_sec: float = 1.5):
        """Add random delay to avoid rate limiting."""
        await asyncio.sleep(random.uniform(min_sec, max_sec))
    
    async def check_browser_available(self) -> bool:
        """Check if Playwright browser is available."""
        if self._browser_available is not None:
            return self._browser_available
        
        try:
            from playwright.async_api import async_playwright
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=True)
            await browser.close()
            await playwright.stop()
            self._browser_available = True
        except Exception:
            self._browser_available = False
        
        return self._browser_available
    
    @asynccontextmanager
    async def get_browser_page(self):
        """Get a Playwright browser page with stealth settings."""
        from playwright.async_api import async_playwright
        
        playwright = await async_playwright().start()
        
        try:
            # Launch browser with safer settings
            browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                ]
            )
            
            # Create context with realistic settings
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent=self.ua.random,
                locale='en-IN',
            )
            
            page = await context.new_page()
            page.set_default_timeout(30000)
            
            try:
                yield page
            finally:
                await context.close()
                await browser.close()
        finally:
            await playwright.stop()
    
    @abstractmethod
    async def search(self, query: str) -> List["ProductResult"]:
        """Search for products on the platform."""
        pass
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get an async HTTP client."""
        return httpx.AsyncClient(
            headers=self.get_headers(),
            timeout=self.timeout,
            follow_redirects=True,
        )
    
    def parse_price(self, price_str: str) -> float:
        """Parse price string to float."""
        if not price_str:
            return 0.0
        # Remove currency symbols, commas, and whitespace
        import re
        cleaned = re.sub(r'[₹,\s]', '', price_str)
        # Extract first number sequence with optional decimal
        match = re.search(r'(\d+(?:\.\d+)?)', cleaned)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.0
    
    async def safe_search(self, query: str) -> List["ProductResult"]:
        """Wrapper for search with error handling."""
        try:
            return await self.search(query)
        except Exception as e:
            print(f"{self.PLATFORM_NAME} search error: {e}")
            return []
