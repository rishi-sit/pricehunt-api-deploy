"""JioMart Quick scraper - for groceries with quick delivery (10-30 mins).

URL: https://www.jiomart.com/search/{query}?tab=groceries
Uses Playwright for browser-based scraping to avoid 403 blocks.
"""
from typing import List
from .base import BaseScraper, ProductResult


class JioMartQuickScraper(BaseScraper):
    """Scraper for JioMart Quick (groceries, 10-30 mins delivery)."""
    
    PLATFORM_NAME = "JioMart Quick"
    BASE_URL = "https://www.jiomart.com"
    
    def __init__(self, pincode: str = "560087"):
        super().__init__(pincode)
        
    async def search(self, query: str) -> List[ProductResult]:
        """Search for products on JioMart Quick (groceries) using browser automation."""
        try:
            return await self._browser_search(query)
        except Exception as e:
            print(f"JioMart Quick search error: {e}")
            return []
    
    async def _browser_search(self, query: str) -> List[ProductResult]:
        """Search using Playwright browser."""
        from playwright.async_api import async_playwright
        
        results = []
        search_url = f"{self.BASE_URL}/search/{query.replace(' ', '%20')}?tab=groceries"
        
        print(f"JioMart Quick: Searching with browser {search_url}")
        
        playwright = await async_playwright().start()
        
        try:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='en-IN',
            )
            
            page = await context.new_page()
            await page.goto(search_url, wait_until='networkidle', timeout=20000)
            await page.wait_for_timeout(2000)
            
            # Extract product data using JavaScript
            products_data = await page.evaluate('''() => {
                const products = [];
                
                // Try to find __NEXT_DATA__ for SSR data
                const nextDataScript = document.querySelector('script#__NEXT_DATA__');
                if (nextDataScript) {
                    try {
                        const data = JSON.parse(nextDataScript.textContent);
                        const pageProps = data?.props?.pageProps || {};
                        const searchData = pageProps.searchData || pageProps.initialData || pageProps.data || {};
                        const productList = searchData.products || [];
                        
                        for (const p of productList.slice(0, 10)) {
                            products.push({
                                name: p.name || p.productName || p.title || '',
                                price: parseFloat(p.selling_price || p.sellingPrice || p.sp || p.price || 0),
                                mrp: parseFloat(p.mrp || p.maximum_retail_price || p.originalPrice || 0),
                                url: p.slug ? 'https://www.jiomart.com/p/' + p.slug : 'https://www.jiomart.com',
                                image: p.image || p.imageUrl || p.image_url || p.thumbnail || '',
                                rating: parseFloat(p.rating || p.averageRating || 0) || null
                            });
                        }
                    } catch (e) {
                        console.error('Next data parse error:', e);
                    }
                }
                
                // Fallback: try to find product cards in DOM
                if (products.length === 0) {
                    const cards = document.querySelectorAll('[class*="product-card"], [class*="ProductCard"], [data-testid="product-card"], .plp-card');
                    cards.forEach(card => {
                        const nameEl = card.querySelector('h3, [class*="name"], [class*="title"]');
                        const priceEl = card.querySelector('[class*="price"], [class*="sp"]');
                        const linkEl = card.querySelector('a[href]');
                        const imgEl = card.querySelector('img');
                        
                        if (nameEl && priceEl) {
                            const priceText = priceEl.innerText.replace(/[^0-9.]/g, '');
                            products.push({
                                name: nameEl.innerText.trim(),
                                price: parseFloat(priceText) || 0,
                                mrp: 0,
                                url: linkEl ? (linkEl.href.startsWith('http') ? linkEl.href : 'https://www.jiomart.com' + linkEl.getAttribute('href')) : 'https://www.jiomart.com',
                                image: imgEl ? (imgEl.src || imgEl.dataset.src) : '',
                                rating: null
                            });
                        }
                    });
                }
                
                return products.slice(0, 10);
            }''')
            
            # Parse extracted data
            for p in products_data:
                if p.get('name') and p.get('price', 0) > 0:
                    original_price = p.get('mrp') if p.get('mrp', 0) > p.get('price', 0) else None
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
                        rating=p.get('rating'),
                        available=True,
                        delivery_time="10-30 mins"
                    ))
            
            print(f"JioMart Quick: Found {len(results)} products")
            
            await context.close()
            await browser.close()
            
        except Exception as e:
            print(f"JioMart Quick browser error: {e}")
        finally:
            await playwright.stop()
            
        return results[:5]
