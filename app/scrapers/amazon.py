"""Amazon India scraper for regular Amazon (not Fresh)."""
from typing import Optional, List
import re
from bs4 import BeautifulSoup
from .base import BaseScraper, ProductResult


class AmazonScraper(BaseScraper):
    """Scraper for regular Amazon India (1-3 days delivery)."""
    
    PLATFORM_NAME = "Amazon"
    BASE_URL = "https://www.amazon.in"
    
    def __init__(self, pincode: str = "560087"):
        super().__init__(pincode)
        
    def get_headers(self) -> dict:
        headers = super().get_headers()
        headers.update({
            "Host": "www.amazon.in",
            "Referer": "https://www.amazon.in/",
        })
        return headers
        
    async def search(self, query: str) -> List[ProductResult]:
        """Search for products on regular Amazon India."""
        results = []
        # Regular Amazon search URL
        search_url = f"{self.BASE_URL}/s?k={query.replace(' ', '+')}&ref=nb_sb_noss"
        
        try:
            async with await self.get_client() as client:
                cookies = {
                    "session-id-time": "2082787201l",
                    "i18n-prefs": "INR",
                }
                
                response = await client.get(search_url, cookies=cookies)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "lxml")
                    products = soup.select('[data-component-type="s-search-result"]')[:15]
                    
                    if not products:
                        products = soup.select('.s-result-item[data-asin]')[:15]
                    
                    for product in products:
                        try:
                            result = self._parse_product(product)
                            if result and result.price > 0:
                                results.append(result)
                        except Exception:
                            continue
                            
        except Exception as e:
            print(f"Amazon search error: {e}")
        
        return results[:5]
    
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
                for selector in ['h2 a span', 'h2 span', '.a-size-medium.a-color-base.a-text-normal']:
                    name_elem = product.select_one(selector)
                    if name_elem:
                        name = name_elem.get_text(strip=True)
                        if name and len(name) > 5:
                            break
            
            if not name or len(name) < 5:
                return None
            
            # Get URL
            link_elem = product.select_one('h2 a')
            url = f"{self.BASE_URL}/dp/{asin}"
            if link_elem and link_elem.get('href'):
                href = link_elem['href']
                url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
            
            # Get price
            price = 0.0
            for selector in ['span.a-price:not(.a-text-price) .a-offscreen', 'span.a-price-whole']:
                price_elem = product.select_one(selector)
                if price_elem:
                    price = self.parse_price(price_elem.get_text())
                    if price > 0:
                        break
            
            # Fallback price extraction
            if price <= 0:
                for span in product.find_all('span'):
                    text = span.get_text(strip=True)
                    if text.startswith('₹') and len(text) < 10:
                        price = self.parse_price(text)
                        if price > 0:
                            break
            
            if price <= 0:
                return None
            
            # Get original price
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
                delivery_time="1-3 days"
            )
            
        except Exception:
            return None
