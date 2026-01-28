"""Zepto scraper using Playwright browser automation."""
from typing import List
import re
from .base import BaseScraper, ProductResult


class ZeptoScraper(BaseScraper):
    """Scraper for Zepto."""
    
    PLATFORM_NAME = "Zepto"
    BASE_URL = "https://www.zeptonow.com"
    
    def __init__(self, pincode: str = "560087"):
        super().__init__(pincode)
        
    async def search(self, query: str) -> List[ProductResult]:
        """Search for products on Zepto."""
        try:
            return await self._browser_search(query)
        except Exception as e:
            print(f"Zepto search error: {e}")
            return []
    
    async def _browser_search(self, query: str) -> List[ProductResult]:
        """Search using Playwright browser."""
        from playwright.async_api import async_playwright
        
        results = []
        search_url = f"{self.BASE_URL}/search?query={query.replace(' ', '%20')}"
        
        playwright = await async_playwright().start()
        
        try:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                locale='en-IN',
            )
            
            page = await context.new_page()
            await page.goto(search_url, wait_until='networkidle', timeout=25000)
            await page.wait_for_timeout(2000)
            
            # Extract product data including URLs using JavaScript
            products_data = await page.evaluate('''() => {
                const products = [];
                // Find all product cards/links
                const productLinks = document.querySelectorAll('a[href*="/pn/"], a[href*="/prn/"], a[href*="/product"]');
                const seen = new Set();
                
                productLinks.forEach(link => {
                    const href = link.getAttribute('href');
                    if (!href || seen.has(href)) return;
                    seen.add(href);
                    
                    // Get text content of the product card
                    const card = link.closest('[class*="product"], [class*="card"]') || link;
                    const text = card.innerText || '';
                    
                    // Try to find image
                    const img = card.querySelector('img');
                    const imageUrl = img ? (img.src || img.dataset.src) : null;
                    
                    products.push({
                        url: href.startsWith('http') ? href : 'https://www.zeptonow.com' + href,
                        text: text,
                        imageUrl: imageUrl
                    });
                });
                
                return products.slice(0, 10);
            }''')
            
            # Parse the extracted products
            results = self._parse_products_with_urls(products_data)
            
            # Fallback: if no products found with URLs, try text parsing
            if not results:
                body_text = await page.evaluate('() => document.body.innerText')
                results = self._parse_products(body_text, query)
            
            await context.close()
            await browser.close()
            
        except Exception as e:
            print(f"Zepto browser error: {e}")
        finally:
            await playwright.stop()
            
        return results[:5]
    
    def _parse_products_with_urls(self, products_data: list) -> List[ProductResult]:
        """Parse products that have URLs extracted."""
        results = []
        
        for product in products_data:
            url = product.get('url', '')
            text = product.get('text', '')
            image_url = product.get('imageUrl')
            
            if not url or not text:
                continue
            
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if len(lines) < 2:
                continue
            
            price = None
            name = None
            quantity = None
            rating = None
            original_price = None
            
            for line in lines:
                # Match price like ₹123
                price_match = re.match(r'^₹(\d+)$', line)
                if price_match:
                    if not price:
                        price = float(price_match.group(1))
                    elif not original_price and float(price_match.group(1)) > price:
                        original_price = float(price_match.group(1))
                    continue
                
                # Match rating
                if re.match(r'^[0-4]\.[0-9]$', line):
                    rating = float(line)
                    continue
                
                # Skip review count
                if re.match(r'^\([\d.]+k?\)$', line):
                    continue
                
                # Skip discount text
                if 'OFF' in line or '%' in line:
                    continue
                
                # Match quantity
                if re.match(r'^\d+\s*(pack|ml|g|kg|L|pc|pcs|gm|units?)', line, re.I):
                    quantity = line
                    continue
                
                # Get product name
                if not name and len(line) > 5 and not line.startswith('₹') and 'ADD' not in line:
                    name = line
            
            if name and price and price > 0:
                full_name = f"{name} ({quantity})" if quantity else name
                
                discount = None
                if original_price and original_price > price:
                    discount = f"{int((original_price - price) / original_price * 100)}% off"
                
                result = ProductResult(
                    name=full_name[:120],
                    price=price,
                    original_price=original_price,
                    discount=discount,
                    platform=self.PLATFORM_NAME,
                    url=url,
                    image_url=image_url,
                    rating=rating,
                    available=True,
                    delivery_time="10-15 mins"
                )
                results.append(result)
        
        return results
    
    def _parse_products(self, body_text: str, query: str) -> List[ProductResult]:
        """Parse products from page text (fallback when URLs not found)."""
        results = []
        
        parts = body_text.split('\nADD\n')
        
        for part in parts:
            if not part.strip():
                continue
            
            lines = [l.strip() for l in part.split('\n') if l.strip()]
            if len(lines) < 2:
                continue
            
            price = None
            name = None
            quantity = None
            rating = None
            
            for line in lines:
                price_match = re.match(r'^₹(\d+)$', line)
                if price_match and not price:
                    price = float(price_match.group(1))
                    continue
                
                if re.match(r'^[0-4]\.[0-9]$', line):
                    rating = float(line)
                    continue
                
                if re.match(r'^\([\d.]+k?\)$', line):
                    continue
                
                if 'OFF' in line:
                    continue
                
                if re.match(r'^\d+\s*(pack|ml|g|kg|L|pc|pcs)', line, re.I):
                    quantity = line
                    continue
                
                if not name and len(line) > 5 and not line.startswith('₹'):
                    name = line
            
            if name and price and price > 0:
                full_name = f"{name} ({quantity})" if quantity else name
                
                result = ProductResult(
                    name=full_name[:120],
                    price=price,
                    original_price=None,
                    discount=None,
                    platform=self.PLATFORM_NAME,
                    url=f"{self.BASE_URL}/search?query={query}",
                    image_url=None,
                    rating=rating,
                    available=True,
                    delivery_time="10-15 mins"
                )
                results.append(result)
        
        return results
