"""Flipkart scraper - uses marketplace=FLIPKART parameter.

URL format: https://www.flipkart.com/search?q={query}&marketplace=FLIPKART
"""
from typing import Optional, List
import re
from bs4 import BeautifulSoup
from .base import BaseScraper, ProductResult


class FlipkartScraper(BaseScraper):
    """Scraper for regular Flipkart (2-4 days delivery) using FLIPKART marketplace."""
    
    PLATFORM_NAME = "Flipkart"
    BASE_URL = "https://www.flipkart.com"
    
    def __init__(self, pincode: str = "560087"):
        super().__init__(pincode)
        
    def get_headers(self) -> dict:
        headers = super().get_headers()
        headers.update({
            "Host": "www.flipkart.com",
            "Referer": "https://www.flipkart.com/",
        })
        return headers
        
    async def search(self, query: str) -> List[ProductResult]:
        """Search for products on regular Flipkart using FLIPKART marketplace."""
        results = []
        # Use marketplace=FLIPKART for regular Flipkart results
        search_url = f"{self.BASE_URL}/search?q={query.replace(' ', '+')}&marketplace=FLIPKART"
        
        print(f"Flipkart: Searching with {search_url}")
        
        try:
            async with await self.get_client() as client:
                response = await client.get(search_url)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "lxml")
                    results = self._parse_products(soup, query)
                    
        except Exception as e:
            print(f"Flipkart search error: {e}")
        
        return results[:5]
    
    def _parse_products(self, soup: BeautifulSoup, query: str) -> List[ProductResult]:
        """Parse products from search results."""
        results = []
        seen_names = set()
        
        containers = soup.select('div[data-id], a._1fQZEK, a.CGtC98, div._1AtVbE a, div._2kHMtA')[:25]
        
        for container in containers:
            try:
                text = container.get_text(' ', strip=True)
                
                # Extract price
                price_matches = re.findall(r'₹\s*([\d,]+)', text)
                if not price_matches:
                    continue
                
                prices = []
                for p in price_matches:
                    try:
                        price_val = float(p.replace(',', ''))
                        if 0 < price_val < 500000:
                            prices.append(price_val)
                    except:
                        pass
                
                if not prices:
                    continue
                
                price = min(prices)
                original_price = max(prices) if len(prices) > 1 and max(prices) > price else None
                
                # Extract name - prefer title attribute or image alt
                name = ""
                title_elem = container.select_one('[title]')
                if title_elem:
                    name = title_elem.get('title', '')
                
                if not name or len(name) < 10:
                    img = container.select_one('img')
                    if img:
                        name = img.get('alt', '')
                
                if not name or len(name) < 5:
                    name = self._extract_name_fallback(container)
                
                if not name or len(name) < 5:
                    continue
                
                name_key = name[:50].lower()
                if name_key in seen_names:
                    continue
                seen_names.add(name_key)
                
                # URL
                url = self._extract_url(container)
                
                # Image
                img = container.select_one('img')
                image_url = img.get('src') or img.get('data-src') if img else None
                
                # Rating
                rating = self._extract_rating(container)
                
                # Discount
                discount = None
                if original_price and original_price > price:
                    discount_pct = int(((original_price - price) / original_price) * 100)
                    discount = f"{discount_pct}% off"
                
                result = ProductResult(
                    name=name[:120],
                    price=price,
                    original_price=original_price,
                    discount=discount,
                    platform=self.PLATFORM_NAME,
                    url=url,
                    image_url=image_url,
                    rating=rating,
                    available=True,
                    delivery_time="2-4 days"
                )
                results.append(result)
                
            except Exception:
                continue
        
        return results
    
    def _extract_name_fallback(self, container) -> str:
        """Fallback method to extract product name."""
        for selector in ['[class*="KzD"]', '[class*="WKT"]', '[class*="wjc"]', '[class*="IRp"]', 'a.s1Q9rs']:
            elem = container.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                if len(text) > 5 and len(text) < 200:
                    return text
        
        for elem in container.find_all(['a', 'div', 'span']):
            text = elem.get_text(strip=True)
            if text and len(text) > 10 and len(text) < 200 and '₹' not in text:
                if not re.match(r'^[\d.]+$', text) and 'off' not in text.lower():
                    return text
        
        return ""
    
    def _extract_url(self, container) -> str:
        link = container if container.name == 'a' else container.select_one('a[href*="/p/"], a[href*="/product/"]')
        if link:
            href = link.get('href', '')
            if href:
                return f"{self.BASE_URL}{href}" if href.startswith('/') else href
        return f"{self.BASE_URL}/search"
    
    def _extract_rating(self, container) -> Optional[float]:
        for selector in ['._3LWZlK', '.XQDdHH', '[class*="rating"]']:
            elem = container.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                try:
                    rating = float(text)
                    if 0 < rating <= 5:
                        return rating
                except:
                    pass
        return None
