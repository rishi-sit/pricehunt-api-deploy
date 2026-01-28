"""
AI-Powered Scraper Service for PriceHunt
Uses Gemini to extract products from raw HTML when standard extraction fails.

This is the FALLBACK for client-side scraping failures.
When Android app's WebView scraping fails, it sends the raw HTML to this endpoint
and Gemini AI extracts products intelligently.
"""
import os
import json
import asyncio
import re
from typing import List, Dict, Optional, Any
import google.generativeai as genai


class AIScraper:
    """
    AI-powered product extraction from raw HTML.
    Uses Gemini's understanding of HTML structure to extract products
    even when traditional scraping fails.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("⚠️ GEMINI_API_KEY not set - AI scraper will be disabled")
            self.model = None
            return
        
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 2.5 Flash for speed
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0.1,
                "top_p": 0.95,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"
            }
        )
        print("✅ AI Scraper initialized (Gemini 2.5 Flash)")
    
    def is_available(self) -> bool:
        return self.model is not None
    
    async def extract_products_from_html(
        self,
        html: str,
        platform: str,
        search_query: str,
        base_url: str
    ) -> Dict[str, Any]:
        """
        Use Gemini to extract products from raw HTML.
        
        Args:
            html: Raw HTML content from the page
            platform: Platform name (e.g., "Zepto", "Blinkit")
            search_query: What the user searched for
            base_url: Base URL for constructing product links
            
        Returns:
            {
                "products": [...],
                "extraction_method": "gemini_ai",
                "confidence": 0.0-1.0
            }
        """
        if not self.is_available():
            return {
                "products": [],
                "extraction_method": "none",
                "error": "Gemini API not available",
                "ai_powered": False
            }
        
        # Preprocess HTML - remove scripts, styles, and limit size
        cleaned_html = self._preprocess_html(html)
        
        if len(cleaned_html) < 500:
            return {
                "products": [],
                "extraction_method": "none",
                "error": "HTML too short or empty",
                "ai_powered": False
            }
        
        prompt = self._build_extraction_prompt(
            cleaned_html, platform, search_query, base_url
        )
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            result = json.loads(response.text)
            
            # Validate and enrich products
            products = self._validate_products(
                result.get("products", []),
                platform,
                base_url
            )
            
            return {
                "products": products,
                "extraction_method": "gemini_ai",
                "confidence": result.get("confidence", 0.7),
                "ai_powered": True,
                "products_found": len(products)
            }
            
        except Exception as e:
            print(f"❌ AI extraction error for {platform}: {e}")
            return {
                "products": [],
                "extraction_method": "none",
                "error": str(e),
                "ai_powered": False
            }
    
    async def extract_from_multiple_platforms(
        self,
        platform_html_list: List[Dict[str, str]],
        search_query: str
    ) -> Dict[str, Any]:
        """
        Extract products from multiple platform HTMLs in parallel.
        
        Args:
            platform_html_list: List of {"platform": str, "html": str, "base_url": str}
            search_query: What the user searched for
            
        Returns:
            {
                "results": {
                    "Zepto": {"products": [...], ...},
                    "Blinkit": {"products": [...], ...}
                },
                "total_products": int
            }
        """
        tasks = []
        for item in platform_html_list:
            task = self.extract_products_from_html(
                html=item.get("html", ""),
                platform=item.get("platform", "Unknown"),
                search_query=search_query,
                base_url=item.get("base_url", "")
            )
            tasks.append((item.get("platform"), task))
        
        results = {}
        total = 0
        
        for platform, task in tasks:
            try:
                result = await task
                results[platform] = result
                total += len(result.get("products", []))
            except Exception as e:
                results[platform] = {
                    "products": [],
                    "error": str(e),
                    "ai_powered": False
                }
        
        return {
            "results": results,
            "total_products": total,
            "ai_powered": True
        }
    
    def _preprocess_html(self, html: str) -> str:
        """Clean HTML for better AI processing"""
        # Remove script tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove style tags
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove comments
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        # Remove excessive whitespace
        html = re.sub(r'\s+', ' ', html)
        # Remove SVG content
        html = re.sub(r'<svg[^>]*>.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove base64 images
        html = re.sub(r'data:image/[^"\']+', 'IMAGE_DATA', html)
        
        # Limit size (Gemini has token limits)
        max_chars = 100000  # ~25k tokens
        if len(html) > max_chars:
            # Try to keep the product-relevant parts
            # Look for common product container patterns
            product_section = self._find_product_section(html)
            if product_section and len(product_section) > 5000:
                html = product_section[:max_chars]
            else:
                html = html[:max_chars]
        
        return html.strip()
    
    def _find_product_section(self, html: str) -> Optional[str]:
        """Try to find the main product listing section"""
        # Common patterns for product containers
        patterns = [
            r'<div[^>]*(?:class|id)=["\'][^"\']*(?:product|search-result|listing|items|grid)[^"\']*["\'][^>]*>.*?(?=<footer|<div[^>]*(?:class|id)=["\'][^"\']*(?:footer|bottom)[^"\']*["\'])',
            r'<main[^>]*>.*?</main>',
            r'<section[^>]*(?:class|id)=["\'][^"\']*(?:product|result)[^"\']*["\'][^>]*>.*?</section>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, flags=re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _build_extraction_prompt(
        self,
        html: str,
        platform: str,
        search_query: str,
        base_url: str
    ) -> str:
        """Build the prompt for Gemini extraction"""
        
        return f"""You are an expert web scraper. Extract ALL product information from this e-commerce HTML.

PLATFORM: {platform}
SEARCH QUERY: "{search_query}"
BASE URL: {base_url}

EXTRACTION RULES:
1. Find ALL products displayed on this page
2. Extract: name, price (in INR ₹), original_price (if discounted), image_url, product_url
3. Price format: Look for ₹ or Rs. followed by numbers (e.g., ₹45, Rs. 120, Rs 99)
4. Product names are usually in img alt text, h2/h3/h4 tags, or link text
5. Image URLs often contain "cdn", "image", "product" in the path
6. Product URLs often contain "/p/", "/product/", "/dp/", "/prn/", "/item/"
7. If product_url is relative (starts with /), prepend the base_url
8. IGNORE: ads, banners, navigation items, footer content, delivery times shown alone

PLATFORM-SPECIFIC HINTS:
- Zepto: Products in cards, prices near images, URLs have /prn/ pattern
- Blinkit: Uses data-* attributes, URLs have /prn/ or /prid/ pattern
- BigBasket: Next.js app, check __NEXT_DATA__ script, URLs have /pd/ pattern
- Instamart/Swiggy: Cloudinary images, nested JSON in scripts
- Flipkart: data-id attributes, ₹ symbol prices, /p/ URL pattern
- JioMart: Product cards with plp-card class, /p/ URL pattern
- Amazon: data-asin attributes, /dp/ URLs

HTML CONTENT:
{html[:80000]}

Return ONLY valid JSON:
{{
    "products": [
        {{
            "name": "Product name (clean, no extra text)",
            "price": 45.00,
            "original_price": 50.00,
            "image_url": "https://...",
            "product_url": "https://...",
            "in_stock": true
        }}
    ],
    "confidence": 0.85,
    "extraction_notes": "Brief notes about extraction quality"
}}

IMPORTANT:
- Return empty products array if no products found
- Price must be a number (not string)
- Confidence: 0.9+ if clear product grid, 0.7-0.9 if some uncertainty, <0.7 if guessing
- Maximum 20 products

JSON only:"""
    
    def _validate_products(
        self,
        products: List[Dict],
        platform: str,
        base_url: str
    ) -> List[Dict]:
        """Validate and enrich extracted products"""
        validated = []
        seen_names = set()
        
        for p in products:
            name = p.get("name", "").strip()
            price = p.get("price")
            
            # Skip invalid products
            if not name or len(name) < 3 or len(name) > 150:
                continue
            if not price or not isinstance(price, (int, float)) or price <= 0:
                continue
            if price > 50000:  # Unrealistic price for groceries
                continue
            
            # Skip duplicates
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)
            
            # Fix URLs
            product_url = p.get("product_url", "")
            if product_url and not product_url.startswith("http"):
                product_url = f"{base_url.rstrip('/')}/{product_url.lstrip('/')}"
            
            image_url = p.get("image_url", "")
            if image_url and not image_url.startswith("http"):
                if image_url.startswith("//"):
                    image_url = f"https:{image_url}"
                else:
                    image_url = f"{base_url.rstrip('/')}/{image_url.lstrip('/')}"
            
            validated.append({
                "name": name,
                "price": float(price),
                "original_price": float(p["original_price"]) if p.get("original_price") else None,
                "image_url": image_url,
                "product_url": product_url or base_url,
                "platform": platform,
                "in_stock": p.get("in_stock", True),
                "ai_extracted": True
            })
        
        return validated[:20]  # Limit to 20 products


# Singleton instance
_ai_scraper: Optional[AIScraper] = None


def get_ai_scraper() -> AIScraper:
    """Get or create the AI scraper singleton"""
    global _ai_scraper
    if _ai_scraper is None:
        _ai_scraper = AIScraper()
    return _ai_scraper
