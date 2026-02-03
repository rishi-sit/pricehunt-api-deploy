"""
AI-Powered Scraper Service for PriceHunt
Uses AI to extract products from raw HTML when standard extraction fails.

Smart Quota Management:
- Primary: Groq (fastest, 6000 req/day free)
- Fallback: Mistral AI (1B tokens/month free)  
- Fallback: Google Gemini
- Auto-switches when quota exceeded, resets daily at midnight UTC

This is the FALLBACK for client-side scraping failures.
When Android app's WebView scraping fails, it sends the raw HTML to this endpoint
and AI extracts products intelligently.
"""
import os
import json
import asyncio
import re
from typing import List, Dict, Optional, Any
import httpx

# Import shared quota tracker from ai_service
from .ai_service import _quota_tracker


class AIScraper:
    """
    AI-powered product extraction from raw HTML.
    Uses AI's understanding of HTML structure to extract products
    even when traditional scraping fails.
    
    Uses shared quota tracker with AIService for coordinated quota management.
    """
    
    PROVIDER_GROQ = "groq"
    PROVIDER_MISTRAL = "mistral"
    PROVIDER_GEMINI = "gemini"
    
    RATE_LIMIT_CODES = {429, 503}
    
    def __init__(self):
        self.temperature = 0.1
        self.top_p = 0.95
        self.max_output_tokens = 8192
        self.request_timeout_s = 90.0
        
        # Use shared quota tracker
        self.quota = _quota_tracker
        
        # Setup all available providers
        self.providers: Dict[str, Dict[str, Any]] = {}
        self._setup_groq()
        self._setup_mistral()
        self._setup_gemini()
        
        self._available = len(self.providers) > 0
        
        if self._available:
            print(f"✅ AI Scraper initialized (providers: {list(self.providers.keys())})")
        else:
            print("⚠️ No AI providers configured - AI scraper disabled")
    
    def _setup_groq(self):
        """Setup Groq provider (fastest)"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return
        self.providers[self.PROVIDER_GROQ] = {
            "api_key": api_key,
            "model": os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
            "base_url": "https://api.groq.com/openai/v1",
            "type": "openai_compatible",
            "supports_json_mode": True  # llama-3.1 supports JSON mode
        }
    
    def _setup_mistral(self):
        """Setup Mistral AI provider"""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return
        self.providers[self.PROVIDER_MISTRAL] = {
            "api_key": api_key,
            "model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            "base_url": "https://api.mistral.ai/v1",
            "type": "openai_compatible",
            "supports_json_mode": True  # Mistral supports JSON mode
        }
    
    def _setup_gemini(self):
        """Setup Google Gemini provider"""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return
        self.providers[self.PROVIDER_GEMINI] = {
            "api_key": api_key,
            "model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "type": "gemini"
        }
    
    def _get_available_providers(self) -> List[str]:
        """Get list of providers that are configured AND not quota-exhausted"""
        priority = [self.PROVIDER_GROQ, self.PROVIDER_MISTRAL, self.PROVIDER_GEMINI]
        available = []
        for p in priority:
            if p in self.providers and self.quota.is_available(p):
                available.append(p)
        return available
    
    @property
    def provider(self) -> Optional[str]:
        """Current primary provider (dynamic based on quota)"""
        available = self._get_available_providers()
        return available[0] if available else None
    
    def is_available(self) -> bool:
        return self._available and len(self._get_available_providers()) > 0
    
    async def _generate_content(self, prompt: str, provider: Optional[str] = None) -> str:
        """Generate content using specified provider"""
        provider = provider or self.provider
        if not provider or provider not in self.providers:
            raise ValueError("No AI provider available")
        
        config = self.providers[provider]
        
        if config["type"] == "openai_compatible":
            return await self._generate_openai_compatible(prompt, config)
        else:
            return await self._generate_gemini(prompt, config)
    
    async def _generate_openai_compatible(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate content using OpenAI-compatible API (Groq, Mistral)"""
        url = f"{config['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert web scraper. Extract product information from HTML. Always respond with valid JSON only, no markdown, no explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_tokens
        }
        
        # Only add response_format for providers/models that support it
        if config.get("supports_json_mode", False):
            payload["response_format"] = {"type": "json_object"}
        
        timeout = httpx.Timeout(timeout=self.request_timeout_s, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""
    
    async def _generate_gemini(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate content using Google Gemini API"""
        model_name = config["model"]
        model_path = model_name if model_name.startswith("models/") else f"models/{model_name}"
        url = f"{config['base_url']}/{model_path}:generateContent"
        params = {"key": config["api_key"]}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json"
            }
        }
        
        timeout = httpx.Timeout(timeout=self.request_timeout_s, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, params=params, json=payload)
            response.raise_for_status()
            data = response.json()
        
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return ""
    
    async def _generate_with_fallback(self, prompt: str) -> tuple[str, str]:
        """Generate content with automatic fallback and quota tracking. Returns (text, provider_used)"""
        providers_to_try = self._get_available_providers()
        
        if not providers_to_try:
            raise Exception("All AI providers exhausted for today")
        
        last_error = None
        
        for provider in providers_to_try:
            try:
                text = await self._generate_content(prompt, provider)
                # Record successful request
                self.quota.record_request(provider)
                return text, provider
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}"
                if e.response.status_code == 429:
                    # Quota exceeded - mark as exhausted for today
                    self.quota.mark_exhausted(provider)
                    print(f"🔄 AI Scraper: {provider} quota exceeded, switching to fallback...")
                    continue
                if e.response.status_code in self.RATE_LIMIT_CODES:
                    print(f"⚠️ AI Scraper: {provider} rate limited, trying fallback...")
                    continue
                print(f"⚠️ AI Scraper: {provider} error: {last_error}, trying fallback...")
                continue
            except Exception as e:
                last_error = str(e)
                print(f"⚠️ AI Scraper: {provider} error: {last_error}, trying fallback...")
                continue
        
        raise Exception(f"All AI providers failed. Last error: {last_error}")
    
    async def extract_products_from_html(
        self,
        html: str,
        platform: str,
        search_query: str,
        base_url: str
    ) -> Dict[str, Any]:
        """
        Use AI to extract products from raw HTML.
        Automatically falls back to next provider on failure.
        
        Args:
            html: Raw HTML content from the page
            platform: Platform name (e.g., "Zepto", "Blinkit")
            search_query: What the user searched for
            base_url: Base URL for constructing product links
            
        Returns:
            {
                "products": [...],
                "extraction_method": "ai",
                "confidence": 0.0-1.0
            }
        """
        if not self.is_available():
            return {
                "products": [],
                "extraction_method": "none",
                "error": "AI API not available",
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
            response_text, provider_used = await self._generate_with_fallback(prompt)
            
            # Parse JSON from response
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start:end + 1]
            
            result = json.loads(cleaned)
            
            # Validate and enrich products
            products = self._validate_products(
                result.get("products", []),
                platform,
                base_url
            )
            
            return {
                "products": products,
                "extraction_method": "ai",
                "provider": provider_used,
                "model": self.providers[provider_used]["model"],
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
5. If size/quantity/pack info is visible (e.g., 500ml, 1kg, pack of 6),
   include it in the product name so the client can compute per-unit price.
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

            quantity_hint = (
                str(p.get("quantity") or p.get("size") or p.get("pack_size") or p.get("packSize") or "")
            ).strip()
            if quantity_hint and quantity_hint.lower() not in name.lower():
                number_match = re.search(r"(\d+(?:\.\d+)?)", quantity_hint)
                if number_match:
                    if float(number_match.group(1)) <= 0:
                        quantity_hint = ""
                if quantity_hint:
                    candidate = f"{name} {quantity_hint}".strip()
                    if len(candidate) <= 150:
                        name = candidate
            
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
