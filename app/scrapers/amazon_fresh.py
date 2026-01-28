"""Amazon Fresh scraper - searches from Amazon Fresh/Now store.

URL format: https://www.amazon.in/s?k={query}&i=nowstore
The 'i=nowstore' parameter filters to Amazon Fresh products only.
"""
from typing import Optional, List
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .base import BaseScraper, ProductResult


class AmazonFreshScraper(BaseScraper):
    """Scraper for Amazon Fresh (2-4 hours delivery) using nowstore index."""
    
    PLATFORM_NAME = "Amazon Fresh"
    BASE_URL = "https://www.amazon.in"
    USE_BROWSER = True
    _executor = ThreadPoolExecutor(max_workers=2)
    
    def __init__(self, pincode: str = "560087"):
        super().__init__(pincode)
        
    async def search(self, query: str) -> List[ProductResult]:
        """Search for products on Amazon Fresh using nowstore."""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self._executor,
                self._sync_browser_search,
                query
            )
            return results[:5]
        except Exception as e:
            print(f"Amazon Fresh search error: {e}")
            return []
    
    def _sync_browser_search(self, query: str) -> List[ProductResult]:
        """Synchronous browser search running in a thread."""
        from playwright.sync_api import sync_playwright
        from bs4 import BeautifulSoup
        
        results = []
        
        # Amazon Fresh search URL with nowstore index
        search_url = f"{self.BASE_URL}/s?k={query.replace(' ', '+')}&i=nowstore"
        
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    locale='en-IN',
                )
                page = context.new_page()
                
                print(f"Amazon Fresh: Searching with URL {search_url}")
                page.goto(search_url, wait_until='domcontentloaded', timeout=20000)
                page.wait_for_timeout(3000)  # Give time for products to load
                
                # Verify we're on nowstore
                current_url = page.url
                if 'nowstore' not in current_url.lower():
                    print(f"Amazon Fresh: Warning - not on nowstore, URL: {current_url}")
                
                # Parse the page
                html = page.content()
                soup = BeautifulSoup(html, 'lxml')
                
                # Find product containers
                products = soup.select('[data-component-type="s-search-result"]')[:15]
                
                if not products:
                    products = soup.select('.s-result-item[data-asin]')[:15]
                
                print(f"Amazon Fresh: Found {len(products)} product containers")
                
                for product in products:
                    try:
                        result = self._parse_product(product)
                        if result and result.price > 0:
                            results.append(result)
                    except Exception as e:
                        continue
                
                context.close()
                browser.close()
                
            except Exception as e:
                print(f"Amazon Fresh browser error: {e}")
        
        return results
    
    def _parse_product(self, product) -> Optional[ProductResult]:
        """Parse a product element from search results."""
        try:
            # Skip sponsored products
            if product.select_one('.s-sponsored-label-info-icon'):
                return None
            
            asin = product.get('data-asin', '')
            if not asin:
                return None
            
            # Get product name - prefer image alt which has full name
            name = ""
            
            # First try image alt text (has full product name)
            img = product.select_one('img.s-image')
            if img:
                name = img.get('alt', '')
            
            # Fallback to aria-label on link
            if not name or len(name) < 10:
                link = product.select_one('h2 a[aria-label]')
                if link:
                    name = link.get('aria-label', '')
            
            # Fallback to h2 text
            if not name or len(name) < 10:
                for selector in ['h2 a span', 'h2 span', '.a-size-medium', '.a-size-base-plus']:
                    name_elem = product.select_one(selector)
                    if name_elem:
                        name = name_elem.get_text(strip=True)
                        if name and len(name) > 5:
                            break
            
            if not name or len(name) < 5:
                return None
            
            # Get URL - append i=nowstore to stay in Amazon Fresh context
            link_elem = product.select_one('h2 a')
            url = f"{self.BASE_URL}/dp/{asin}?i=nowstore"
            if link_elem and link_elem.get('href'):
                href = link_elem['href']
                base_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                # Add nowstore parameter
                if '?' in base_url:
                    url = f"{base_url}&i=nowstore"
                else:
                    url = f"{base_url}?i=nowstore"
            
            # Get price - try multiple selectors
            price = 0.0
            for selector in [
                'span.a-price:not(.a-text-price) .a-offscreen',
                'span.a-price-whole',
                '.a-price .a-offscreen',
                'span[data-a-color="price"] .a-offscreen'
            ]:
                price_elem = product.select_one(selector)
                if price_elem:
                    price = self.parse_price(price_elem.get_text())
                    if price > 0:
                        break
            
            # Fallback price extraction from any span with ₹
            if price <= 0:
                for span in product.find_all('span'):
                    text = span.get_text(strip=True)
                    if text.startswith('₹') and len(text) < 15:
                        price = self.parse_price(text)
                        if price > 0:
                            break
            
            if price <= 0:
                return None
            
            # Get original price (MRP)
            original_price = None
            orig_elem = product.select_one('.a-price.a-text-price .a-offscreen')
            if orig_elem:
                orig = self.parse_price(orig_elem.get_text())
                if orig > price:
                    original_price = orig
            
            # Discount
            discount = None
            if original_price and original_price > price:
                discount_pct = int(((original_price - price) / original_price) * 100)
                discount = f"{discount_pct}% off"
            
            # Image
            image_elem = product.select_one('img.s-image')
            image_url = image_elem.get('src') if image_elem else None
            
            # Rating
            rating = None
            rating_elem = product.select_one('span.a-icon-alt')
            if rating_elem:
                try:
                    rating = float(rating_elem.get_text().split()[0])
                except:
                    pass
            
            return ProductResult(
                name=name[:120],
                price=price,
                original_price=original_price,
                discount=discount,
                platform=self.PLATFORM_NAME,
                url=url,
                image_url=image_url,
                rating=rating,
                available=True,
                delivery_time="2-4 hours"
            )
            
        except Exception:
            return None
