"""Flipkart Minutes scraper - uses marketplace=HYPERLOCAL.

Searches from Flipkart Minutes store after setting location.
URL after search: flipkart.com/search?q=...&marketplace=HYPERLOCAL
"""
from typing import Optional, List
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .base import BaseScraper, ProductResult


class FlipkartMinutesScraper(BaseScraper):
    """Scraper for Flipkart Minutes (6-10 mins delivery) using HYPERLOCAL marketplace."""
    
    PLATFORM_NAME = "Flipkart Minutes"
    BASE_URL = "https://www.flipkart.com"
    MINUTES_STORE_URL = "https://www.flipkart.com/flipkart-minutes-store"
    USE_BROWSER = True
    _executor = ThreadPoolExecutor(max_workers=2)
    
    # Bangalore coordinates
    BANGALORE_LAT = 12.9716
    BANGALORE_LON = 77.5946
    
    def __init__(self, pincode: str = "560087"):
        super().__init__(pincode)
        
    async def search(self, query: str) -> List[ProductResult]:
        """Search for products on Flipkart Minutes."""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self._executor,
                self._sync_browser_search,
                query
            )
            return results[:5]
        except Exception as e:
            print(f"Flipkart Minutes search error: {e}")
            return []
    
    def _sync_browser_search(self, query: str) -> List[ProductResult]:
        """Synchronous browser search."""
        from playwright.sync_api import sync_playwright
        from bs4 import BeautifulSoup
        
        results = []
        
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    locale='en-IN',
                    geolocation={'latitude': self.BANGALORE_LAT, 'longitude': self.BANGALORE_LON},
                    permissions=['geolocation']
                )
                page = context.new_page()
                
                # Step 1: Go to Flipkart
                print("Flipkart Minutes: Going to Flipkart...")
                page.goto("https://www.flipkart.com", wait_until='domcontentloaded', timeout=15000)
                page.wait_for_timeout(1500)
                
                # Close popup
                try:
                    page.click('button._2KpZ6l._2doB4z', timeout=2000)
                except:
                    pass
                
                # Step 2: Click Minutes and set location
                print("Flipkart Minutes: Setting location via Minutes store...")
                try:
                    page.click('text="Minutes"', timeout=5000)
                    page.wait_for_timeout(2000)
                    page.click('text="Use my current location"', timeout=5000)
                    page.wait_for_timeout(3000)
                    
                    try:
                        page.click('text=/Confirm|Continue/i', timeout=2000)
                        page.wait_for_timeout(2000)
                    except:
                        pass
                except Exception as e:
                    print(f"Flipkart Minutes: Location setup failed: {e}")
                    context.close()
                    browser.close()
                    return []
                
                # Verify we're on Minutes store
                if 'minutes-store' not in page.url.lower() and 'HYPERLOCAL' not in page.url.upper():
                    print(f"Flipkart Minutes: Not on store, URL: {page.url}")
                    context.close()
                    browser.close()
                    return []
                
                # Step 3: Search using the search box (NOT direct URL navigation)
                search_input = page.query_selector('input[name="q"], input[placeholder*="Search"]')
                if search_input:
                    print(f"Flipkart Minutes: Searching for '{query}'...")
                    search_input.fill(query)
                    search_input.press("Enter")
                    page.wait_for_timeout(3000)
                else:
                    print("Flipkart Minutes: No search input found")
                    context.close()
                    browser.close()
                    return []
                
                # Verify HYPERLOCAL marketplace
                if 'HYPERLOCAL' in page.url.upper():
                    print("Flipkart Minutes: ✓ On HYPERLOCAL marketplace")
                else:
                    print(f"Flipkart Minutes: Warning - URL: {page.url}")
                
                # Step 4: Parse results
                html = page.content()
                body_text = page.evaluate('() => document.body.innerText')
                
                soup = BeautifulSoup(html, 'lxml')
                results = self._parse_products(soup, body_text)
                
                print(f"Flipkart Minutes: Found {len(results)} products")
                
                context.close()
                browser.close()
                
            except Exception as e:
                print(f"Flipkart Minutes browser error: {e}")
        
        return results
    
    def _parse_products(self, soup, body_text: str) -> List[ProductResult]:
        """Parse products from page."""
        results = []
        seen_names = set()
        
        # First, collect all product links with their parent containers
        product_links = soup.select('a[href*="/p/"]')[:25]
        
        for link in product_links:
            try:
                href = link.get('href', '')
                if not href or '/p/' not in href:
                    continue
                
                # Build the full URL
                if href.startswith('/'):
                    url = f"{self.BASE_URL}{href}"
                else:
                    url = href
                
                # Only add marketplace=HYPERLOCAL if not already present (case-insensitive check)
                if 'MARKETPLACE=HYPERLOCAL' not in url.upper():
                    if '?' in url:
                        url = f"{url}&marketplace=HYPERLOCAL"
                    else:
                        url = f"{url}?marketplace=HYPERLOCAL"
                
                # Get parent container for more info
                container = link.parent
                if container:
                    container = container.parent or container
                
                text = link.get_text(' ', strip=True) if link else ''
                parent_text = container.get_text(' ', strip=True) if container else text
                
                # Extract price from parent
                price_matches = re.findall(r'₹\s*([\d,]+)', parent_text)
                if not price_matches:
                    continue
                
                prices = [float(p.replace(',', '')) for p in price_matches if 0 < float(p.replace(',', '')) < 50000]
                if not prices:
                    continue
                
                price = min(prices)
                original_price = max(prices) if len(prices) > 1 and max(prices) > price else None
                
                # Get name from title, alt, or aria-label
                name = ""
                title_elem = link.select_one('[title]') or link
                if title_elem:
                    name = title_elem.get('title', '') or title_elem.get('aria-label', '')
                
                if not name or len(name) < 10:
                    img = link.select_one('img') or (container.select_one('img') if container else None)
                    if img:
                        name = img.get('alt', '')
                
                if not name or len(name) < 5:
                    # Extract from link text
                    for elem in link.find_all(['div', 'span']):
                        t = elem.get_text(strip=True)
                        if t and len(t) > 10 and len(t) < 150 and '₹' not in t and '%' not in t:
                            name = t
                            break
                
                if not name or len(name) < 5:
                    continue
                
                name_key = name[:40].lower()
                if name_key in seen_names:
                    continue
                seen_names.add(name_key)
                
                # Image
                img = link.select_one('img') or (container.select_one('img') if container else None)
                image_url = img.get('src') or img.get('data-src') if img else None
                
                discount = None
                if original_price and original_price > price:
                    discount = f"{int(((original_price - price) / original_price) * 100)}% off"
                
                results.append(ProductResult(
                    name=name[:120],
                    price=price,
                    original_price=original_price,
                    discount=discount,
                    platform=self.PLATFORM_NAME,
                    url=url,
                    image_url=image_url,
                    rating=None,
                    available=True,
                    delivery_time="6-10 mins"
                ))
                
                if len(results) >= 5:
                    break
                    
            except:
                continue
        
        # Fallback to container-based parsing if link parsing didn't work
        if len(results) < 3:
            containers = soup.select('div[data-id], div._1AtVbE, div._2kHMtA, div._4ddWXP')[:25]
            
            for container in containers:
                try:
                    text = container.get_text(' ', strip=True)
                    
                    price_matches = re.findall(r'₹\s*([\d,]+)', text)
                    if not price_matches:
                        continue
                    
                    prices = []
                    for p in price_matches:
                        try:
                            val = float(p.replace(',', ''))
                            if 0 < val < 50000:
                                prices.append(val)
                        except:
                            pass
                    
                    if not prices:
                        continue
                    
                    price = min(prices)
                    original_price = max(prices) if len(prices) > 1 and max(prices) > price else None
                    
                    # Get name
                    name = ""
                    title_elem = container.select_one('[title]')
                    if title_elem:
                        name = title_elem.get('title', '')
                    
                    if not name or len(name) < 10:
                        img = container.select_one('img')
                        if img:
                            name = img.get('alt', '')
                    
                    if not name or len(name) < 5:
                        for elem in container.find_all(['a', 'div', 'span']):
                            t = elem.get_text(strip=True)
                            if t and len(t) > 10 and len(t) < 150 and '₹' not in t and '%' not in t:
                                name = t
                                break
                    
                    if not name or len(name) < 5:
                        continue
                    
                    name_key = name[:40].lower()
                    if name_key in seen_names:
                        continue
                    seen_names.add(name_key)
                    
                    # URL - get from link if available
                    link = container if container.name == 'a' else container.select_one('a[href*="/p/"]')
                    url = self.MINUTES_STORE_URL
                    if link:
                        href = link.get('href', '')
                        if href:
                            url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                            # Only add marketplace if not already present (case-insensitive)
                            if 'MARKETPLACE=HYPERLOCAL' not in url.upper():
                                if '?' in url:
                                    url = f"{url}&marketplace=HYPERLOCAL"
                                else:
                                    url = f"{url}?marketplace=HYPERLOCAL"
                    
                    # Image
                    img = container.select_one('img')
                    image_url = img.get('src') or img.get('data-src') if img else None
                    
                    discount = None
                    if original_price and original_price > price:
                        discount = f"{int(((original_price - price) / original_price) * 100)}% off"
                    
                    results.append(ProductResult(
                        name=name[:120],
                        price=price,
                        original_price=original_price,
                        discount=discount,
                        platform=self.PLATFORM_NAME,
                        url=url,
                        image_url=image_url,
                        rating=None,
                        available=True,
                        delivery_time="6-10 mins"
                    ))
                    
                except:
                    continue
        
        # Fallback to text parsing if HTML didn't work
        if len(results) < 3:
            text_results = self._parse_from_text(body_text, seen_names)
            results.extend(text_results)
        
        return results[:5]
    
    def _parse_from_text(self, body_text: str, seen_names: set) -> List[ProductResult]:
        """Parse from plain text as fallback."""
        results = []
        lines = body_text.split('\n')
        
        i = 0
        while i < len(lines) - 3 and len(results) < 5:
            line = lines[i].strip()
            
            # Look for product-like lines
            if len(line) > 15 and len(line) < 120 and '₹' not in line and '%' not in line:
                for j in range(i + 1, min(i + 5, len(lines))):
                    price_match = re.search(r'₹\s*([\d,]+)', lines[j])
                    if price_match:
                        price = float(price_match.group(1).replace(',', ''))
                        if 0 < price < 5000:
                            name_key = line[:40].lower()
                            if name_key not in seen_names:
                                seen_names.add(name_key)
                                results.append(ProductResult(
                                    name=line[:120],
                                    price=price,
                                    original_price=None,
                                    discount=None,
                                    platform=self.PLATFORM_NAME,
                                    url=self.MINUTES_STORE_URL,
                                    image_url=None,
                                    rating=None,
                                    available=True,
                                    delivery_time="6-10 mins"
                                ))
                            break
            i += 1
        
        return results
