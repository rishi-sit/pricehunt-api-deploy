"""BigBasket scraper for grocery products.

Uses Playwright for browser-based scraping to bypass anti-bot protection.
"""
from typing import List
from .base import BaseScraper, ProductResult


class BigBasketScraper(BaseScraper):
    """Scraper for BigBasket grocery delivery."""
    
    PLATFORM_NAME = "BigBasket"
    BASE_URL = "https://www.bigbasket.com"
    
    def __init__(self, pincode: str = "560087"):
        super().__init__(pincode)
        
    async def search(self, query: str) -> List[ProductResult]:
        """Search for products on BigBasket using browser automation."""
        try:
            return await self._browser_search(query)
        except Exception as e:
            print(f"BigBasket search error: {e}")
            return []
    
    async def _browser_search(self, query: str) -> List[ProductResult]:
        """Search using Playwright browser."""
        from playwright.async_api import async_playwright
        
        results = []
        search_url = f"{self.BASE_URL}/ps/?q={query.replace(' ', '%20')}"
        
        print(f"BigBasket: Searching with browser {search_url}")
        
        playwright = await async_playwright().start()
        
        try:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='en-IN',
            )
            
            # Set location cookie
            await context.add_cookies([{
                'name': '_bb_pin_code',
                'value': self.pincode,
                'domain': '.bigbasket.com',
                'path': '/'
            }])
            
            page = await context.new_page()
            await page.goto(search_url, wait_until='networkidle', timeout=20000)
            await page.wait_for_timeout(3000)  # Wait for products to load
            
            # Extract product data using JavaScript
            products_data = await page.evaluate('''() => {
                const products = [];
                
                // BigBasket uses a variety of selectors for product cards
                const selectors = [
                    '[data-qa="product"]',
                    '[class*="PaginateItems"] > li',
                    '.product-card',
                    '[class*="ProductCard"]',
                    'li[class*="product"]',
                    '.prod-deck',
                    '[class*="SKUDeck"]',
                    '[class*="ProductListing"] > div'
                ];
                
                let cards = [];
                for (const selector of selectors) {
                    cards = document.querySelectorAll(selector);
                    if (cards.length > 0) break;
                }
                
                // If no cards found, try finding elements with price-like text
                if (cards.length === 0) {
                    const allElements = document.querySelectorAll('div, li, article');
                    cards = Array.from(allElements).filter(el => {
                        const text = el.innerText || '';
                        return text.includes('₹') && text.length < 500;
                    }).slice(0, 20);
                }
                
                cards.forEach(card => {
                    try {
                        // Get name
                        let name = '';
                        const nameSelectors = [
                            '[data-qa="product-title"]',
                            'h3',
                            '[class*="ProductName"]',
                            '[class*="product-name"]',
                            '[class*="ItemName"]',
                            'a[title]'
                        ];
                        for (const sel of nameSelectors) {
                            const el = card.querySelector(sel);
                            if (el) {
                                name = el.innerText?.trim() || el.getAttribute('title') || '';
                                if (name && name.length > 3) break;
                            }
                        }
                        
                        // Get price
                        let price = 0;
                        const priceSelectors = [
                            '[data-qa="product-price"]',
                            '[class*="discnt-price"]',
                            '[class*="sale-price"]',
                            '[class*="SalePrice"]',
                            '[class*="Price"]:not([class*="mrp"])',
                            'span[class*="price"]'
                        ];
                        for (const sel of priceSelectors) {
                            const el = card.querySelector(sel);
                            if (el) {
                                const priceText = el.innerText.replace(/[^0-9.]/g, '');
                                price = parseFloat(priceText) || 0;
                                if (price > 0) break;
                            }
                        }
                        
                        // Fallback: find ₹ in text
                        if (price === 0) {
                            const match = card.innerText.match(/₹\s*(\d+(?:\.\d+)?)/);
                            if (match) price = parseFloat(match[1]);
                        }
                        
                        // Get original price (MRP)
                        let mrp = 0;
                        const mrpSelectors = ['.mrp-price', '[class*="MRP"]', 'del', 's', '[class*="strikethrough"]'];
                        for (const sel of mrpSelectors) {
                            const el = card.querySelector(sel);
                            if (el) {
                                const mrpText = el.innerText.replace(/[^0-9.]/g, '');
                                mrp = parseFloat(mrpText) || 0;
                                if (mrp > 0) break;
                            }
                        }
                        
                        // Get URL
                        let url = 'https://www.bigbasket.com';
                        const linkEl = card.querySelector('a[href*="/pd/"]') || card.querySelector('a[href]');
                        if (linkEl) {
                            const href = linkEl.getAttribute('href');
                            url = href.startsWith('http') ? href : 'https://www.bigbasket.com' + href;
                        }
                        
                        // Get image
                        let image = '';
                        const imgEl = card.querySelector('img');
                        if (imgEl) {
                            image = imgEl.src || imgEl.dataset.src || '';
                        }
                        
                        if (name && name.length > 3 && price > 0) {
                            products.push({
                                name: name,
                                price: price,
                                mrp: mrp > price ? mrp : 0,
                                url: url,
                                image: image
                            });
                        }
                    } catch (e) {
                        // Skip errored cards
                    }
                });
                
                return products.slice(0, 10);
            }''')
            
            # Parse extracted data
            for p in products_data:
                if p.get('name') and p.get('price', 0) > 0:
                    original_price = p.get('mrp') if p.get('mrp', 0) > 0 else None
                    discount = None
                    if original_price:
                        discount = f"{int((original_price - p['price']) / original_price * 100)}% off"
                    
                    results.append(ProductResult(
                        name=p['name'][:120],
                        price=p['price'],
                        original_price=original_price,
                        discount=discount,
                        platform=self.PLATFORM_NAME,
                        url=p.get('url', self.BASE_URL),
                        image_url=p.get('image'),
                        rating=None,
                        available=True,
                        delivery_time="2-4 hours"
                    ))
            
            print(f"BigBasket: Found {len(results)} products")
            
            await context.close()
            await browser.close()
            
        except Exception as e:
            print(f"BigBasket browser error: {e}")
        finally:
            await playwright.stop()
            
        return results[:5]
