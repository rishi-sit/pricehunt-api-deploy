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
import time
from typing import List, Dict, Optional, Any
import httpx

# Import shared quota tracker from ai_service
from .ai_service import _quota_tracker

# Import analytics for logging AI processing
from .analytics import AIProcessingLogRequest, log_ai_processing


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
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "base_url": "https://api.groq.com/openai/v1",
            "type": "openai_compatible",
            "supports_json_mode": False  # Disable to debug HTTP 400
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
        """Setup Google Gemini provider with model fallback"""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return
        # HARDCODED model priority: try gemini-2.5-flash first, then lite, then gemma
        models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemma-3-27b-it"]
        self.providers[self.PROVIDER_GEMINI] = {
            "api_key": api_key,
            "model": models[0],
            "models": models,  # All models for fallback
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "type": "gemini"
        }
        # Track unavailable/exhausted models for the day
        self._gemini_unavailable_models: Dict[str, str] = {}
        self._gemini_exhausted_models: Dict[str, str] = {}
        self._gemini_last_reset_date: str = ""
    
    def _get_available_providers(self) -> List[str]:
        """Get list of providers that are configured AND not quota-exhausted"""
        # Priority order: Gemini > Groq > Mistral
        priority = [self.PROVIDER_GEMINI, self.PROVIDER_GROQ, self.PROVIDER_MISTRAL]
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
    
    def force_reset_quota(self) -> Dict[str, Any]:
        """Force reset all quota tracking for AI Scraper"""
        # Reset global quota tracker (shared with AIService)
        quota_result = self.quota.force_reset()
        
        # Reset Gemini model-level tracking
        old_state = {
            "gemini_models_unavailable": list(getattr(self, '_gemini_unavailable_models', {}).keys()),
            "gemini_models_exhausted": list(getattr(self, '_gemini_exhausted_models', {}).keys())
        }
        self._gemini_unavailable_models = {}
        self._gemini_exhausted_models = {}
        self._gemini_last_reset_date = self._get_today()
        
        print(f"🔄 AI Scraper: FORCED quota reset - all models now available")
        
        return {
            "message": "AI Scraper quota reset successful",
            "quota_tracker_reset": quota_result,
            "previous_state": old_state,
            "is_available": self.is_available()
        }

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
    
    def _get_today(self) -> str:
        """Get today's date string for tracking"""
        import datetime
        return datetime.date.today().isoformat()
    
    def _reset_gemini_models_if_new_day(self):
        """Reset model tracking on new day"""
        today = self._get_today()
        if hasattr(self, '_gemini_last_reset_date') and self._gemini_last_reset_date != today:
            self._gemini_unavailable_models = {}
            self._gemini_exhausted_models = {}
            self._gemini_last_reset_date = today
    
    def _is_gemini_model_available(self, model: str) -> bool:
        """Check if a specific Gemini model is available today"""
        self._reset_gemini_models_if_new_day()
        unavailable = getattr(self, '_gemini_unavailable_models', {})
        exhausted = getattr(self, '_gemini_exhausted_models', {})
        return model not in unavailable and model not in exhausted
    
    def _mark_gemini_model_unavailable(self, model: str):
        """Mark a Gemini model as unavailable (404/403 errors)"""
        if not hasattr(self, '_gemini_unavailable_models'):
            self._gemini_unavailable_models = {}
        self._gemini_unavailable_models[model] = self._get_today()
        print(f"⚠️ AI Scraper: Gemini model {model} marked unavailable for today")
    
    def _mark_gemini_model_exhausted(self, model: str):
        """Mark a Gemini model as quota exhausted (429 errors)"""
        if not hasattr(self, '_gemini_exhausted_models'):
            self._gemini_exhausted_models = {}
        self._gemini_exhausted_models[model] = self._get_today()
        print(f"🔄 AI Scraper: Gemini model {model} quota exhausted for today")
    
    async def _generate_gemini(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate content using Google Gemini API with model-level fallback"""
        models = config.get("models", [config["model"]])
        models_to_try = [m for m in models if self._is_gemini_model_available(m)]
        
        if not models_to_try:
            raise Exception("All Gemini models exhausted or unavailable for today")
        
        last_error = None
        for model_name in models_to_try:
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
            
            try:
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
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                last_error = f"HTTP {status}"
                if status == 429:
                    self._mark_gemini_model_exhausted(model_name)
                    continue
                if status in {403, 404}:
                    self._mark_gemini_model_unavailable(model_name)
                    continue
                # Other errors, try next model
                print(f"⚠️ AI Scraper: Gemini model {model_name} error: {last_error}, trying next model...")
                continue
            except Exception as e:
                last_error = str(e)
                print(f"⚠️ AI Scraper: Gemini model {model_name} error: {last_error}, trying next model...")
                continue
        
        raise Exception(f"All Gemini models failed. Last error: {last_error}")
    
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
        base_url: str,
        device_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use AI to extract products from raw HTML.
        Automatically falls back to next provider on failure.
        
        Args:
            html: Raw HTML content from the page
            platform: Platform name (e.g., "Zepto", "Blinkit")
            search_query: What the user searched for
            base_url: Base URL for constructing product links
            device_id: Device identifier for analytics tracking
            
        Returns:
            {
                "products": [...],
                "extraction_method": "ai",
                "confidence": 0.0-1.0
            }
        """
        start_time = time.time()
        input_html_size_kb = len(html) / 1024
        
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
            
            latency_ms = int((time.time() - start_time) * 1000)
            model_name = self.providers[provider_used]["model"]
            
            # Log AI processing analytics
            if device_id:
                try:
                    log_ai_processing(AIProcessingLogRequest(
                        device_id=device_id,
                        search_query=search_query,
                        platform=platform,
                        ai_provider=provider_used,
                        ai_model=model_name,
                        input_html_size_kb=input_html_size_kb,
                        products_found=len(products),
                        products_filtered=len(products),  # All products are relevant at this stage
                        latency_ms=latency_ms,
                        success=True
                    ))
                except Exception as log_err:
                    print(f"⚠️ Failed to log AI analytics: {log_err}")
            
            return {
                "products": products,
                "extraction_method": "ai",
                "provider": provider_used,
                "model": model_name,
                "confidence": result.get("confidence", 0.7),
                "ai_powered": True,
                "products_found": len(products),
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Log failed AI processing
            if device_id:
                try:
                    log_ai_processing(AIProcessingLogRequest(
                        device_id=device_id,
                        search_query=search_query,
                        platform=platform,
                        ai_provider="unknown",
                        ai_model="unknown",
                        input_html_size_kb=input_html_size_kb,
                        products_found=0,
                        products_filtered=0,
                        latency_ms=latency_ms,
                        fallback_reason=str(e),
                        success=False
                    ))
                except Exception as log_err:
                    print(f"⚠️ Failed to log AI analytics: {log_err}")
            
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
        search_query: str,
        device_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract products from multiple platform HTMLs in parallel.
        
        Args:
            platform_html_list: List of {"platform": str, "html": str, "base_url": str}
            search_query: What the user searched for
            device_id: Device identifier for analytics tracking
            
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
                base_url=item.get("base_url", ""),
                device_id=device_id
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
        """Clean HTML for better AI processing while PRESERVING important JSON data"""
        
        # CRITICAL: Extract and preserve JSON data from important script tags FIRST
        preserved_json = []
        
        # Pattern 1: __NEXT_DATA__ (Next.js apps like BigBasket)
        next_data_match = re.search(r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', html, flags=re.DOTALL | re.IGNORECASE)
        if next_data_match:
            preserved_json.append(f"__NEXT_DATA_JSON__: {next_data_match.group(1)[:30000]}")
        
        # Pattern 2: application/ld+json (Schema.org product data)
        ld_json_matches = re.findall(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, flags=re.DOTALL | re.IGNORECASE)
        for i, match in enumerate(ld_json_matches[:3]):  # Limit to 3
            preserved_json.append(f"__LD_JSON_{i}__: {match[:10000]}")
        
        # Pattern 3: Preloaded state (React/Redux apps)
        preload_patterns = [
            r'window\.__PRELOADED_STATE__\s*=\s*(\{.*?\});',
            r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\});',
            r'window\.__DATA__\s*=\s*(\{.*?\});',
        ]
        for pattern in preload_patterns:
            match = re.search(pattern, html, flags=re.DOTALL)
            if match:
                preserved_json.append(f"__PRELOADED_STATE__: {match.group(1)[:20000]}")
                break
        
        # Pattern 4: Look for inline JSON with products/items array
        product_json_patterns = [
            r'"products"\s*:\s*\[([^\]]+(?:\[[^\]]*\][^\]]*)*)\]',
            r'"items"\s*:\s*\[([^\]]+(?:\[[^\]]*\][^\]]*)*)\]',
            r'"searchResults"\s*:\s*\[([^\]]+(?:\[[^\]]*\][^\]]*)*)\]',
            r'"resultsList"\s*:\s*\[([^\]]+(?:\[[^\]]*\][^\]]*)*)\]',
        ]
        for pattern in product_json_patterns:
            product_json_match = re.search(pattern, html, flags=re.DOTALL)
            if product_json_match and len(product_json_match.group(0)) > 100:
                preserved_json.append(f"__PRODUCTS_JSON__: [{product_json_match.group(1)[:20000]}]")
                break
        
        # Pattern 5: JioMart/Magento specific - look for Magento JSON structures
        magento_patterns = [
            r'var\s+mageConfig\s*=\s*(\{.*?\});',
            r'window\.jiomart\s*=\s*(\{.*?\});',
            r'"productInfo"\s*:\s*(\{[^}]+\})',
            r'"plpProductInfo"\s*:\s*(\[.*?\])',
        ]
        for pattern in magento_patterns:
            match = re.search(pattern, html, flags=re.DOTALL)
            if match and len(match.group(0)) > 50:
                preserved_json.append(f"__MAGENTO_DATA__: {match.group(1)[:15000]}")
                break
        
        # Now remove non-essential script tags (keep style for some context)
        # Remove inline event handler scripts
        html = re.sub(r'<script[^>]*>(?:(?!__NEXT_DATA__|application/ld\+json).)*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove style tags
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove comments
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        # Remove SVG content (usually icons)
        html = re.sub(r'<svg[^>]*>.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove base64 images (huge and useless for extraction)
        html = re.sub(r'data:image/[^"\']+', 'IMG_DATA', html)
        # Remove excessive whitespace
        html = re.sub(r'\s+', ' ', html)
        
        # Combine preserved JSON with cleaned HTML
        json_section = "\n\n=== PRESERVED JSON DATA (IMPORTANT - CHECK FOR PRODUCTS) ===\n" + "\n".join(preserved_json) if preserved_json else ""
        
        # Limit total size
        max_html_chars = 70000
        max_json_chars = 30000
        
        if len(html) > max_html_chars:
            # Try to keep the product-relevant parts
            product_section = self._find_product_section(html)
            if product_section and len(product_section) > 5000:
                html = product_section[:max_html_chars]
            else:
                html = html[:max_html_chars]
        
        if len(json_section) > max_json_chars:
            json_section = json_section[:max_json_chars]
        
        return (html + json_section).strip()
    
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
        
        return f"""You are an expert web scraper specializing in Indian e-commerce grocery platforms. Extract ALL product information from this HTML.

PLATFORM: {platform}
SEARCH QUERY: "{search_query}"
BASE URL: {base_url}

CRITICAL: LOOK FOR DATA IN MULTIPLE PLACES:
1. **JSON DATA IN SCRIPTS** (Most Important for SPAs):
   - Look for <script id="__NEXT_DATA__"> containing JSON with product arrays
   - Look for <script type="application/ld+json"> with product schema
   - Look for window.__PRELOADED_STATE__, window.__INITIAL_STATE__, or similar
   - Look for data attributes like data-testid, data-product-id, data-item
   - Search for patterns: "products":[, "items":[, "searchResults":[, "widgets":[

2. **VISIBLE HTML ELEMENTS**:
   - Product cards in div/article elements with class containing: product, item, card, plp, search-result
   - Prices in span/div with class containing: price, mrp, amount, cost, rupee
   - Product names in h1/h2/h3/h4/a/span or img alt text
   - Look for ₹, Rs., Rs, INR followed by numbers

3. **PLATFORM-SPECIFIC PATTERNS**:
   - **Zepto**: JSON in scripts, product cards, /prn/ URLs, prices in "₹XX" format
   - **Blinkit**: data-* attributes, product-card class, /prn/ or /prid/ URLs
   - **BigBasket**: __NEXT_DATA__ script has all products, /pd/ URLs, "products" array in JSON
   - **Instamart/Swiggy**: Cloudinary image URLs (res.cloudinary.com), nested "widgets" in JSON
   - **Flipkart/Flipkart Minutes**: data-id attributes, _FSPP_ scripts, /p/ URLs
   - **JioMart/JioMart Quick**: Look for plp-card, product-card-wrap, product-grid classes. URLs like /p/XXXXX or /pd/XXXXX. Images from jiomartmedia.com or jiomart.com. Prices in "₹XX" or "Rs. XX" format. Look for product-name, product-title, offer-price, mrp-price classes.
   - **Amazon/Amazon Fresh**: data-asin attributes, /dp/ URLs, s-result-item class

EXTRACTION RULES:
1. Extract: name, price (INR), original_price (if discounted), image_url, product_url
2. Price MUST be a number (not string), e.g., 45.00 not "₹45"
3. Clean product names - remove extra whitespace, "Add to Cart", etc.
4. For relative URLs (starting with /), prepend: {base_url}
5. QUANTITY: Extract from name (e.g., "500ml", "1kg", "6 pcs")
   - Format: {{"value": 500, "unit": "ml"}} or {{"value": 6, "unit": "pcs"}}
6. PRICE_PER_UNIT: Calculate if quantity exists
   - Weight/Volume: normalize to per 100g or 100ml
   - Count: price per piece

IGNORE: ads, sponsored banners, delivery info, navigation, footers, login prompts

HTML CONTENT (may contain embedded JSON):
{html[:80000]}

Return ONLY valid JSON (no markdown, no explanation):
{{
    "products": [
        {{
            "name": "Fresh Banana 6 pcs",
            "price": 45.00,
            "original_price": 50.00,
            "image_url": "https://cdn.example.com/banana.jpg",
            "product_url": "https://example.com/p/banana",
            "in_stock": true,
            "quantity": {{"value": 6, "unit": "pcs"}},
            "price_per_unit": 7.5,
            "price_per_unit_display": "₹7.50/pc"
        }}
    ],
    "confidence": 0.85,
    "extraction_notes": "Found products in __NEXT_DATA__ JSON"
}}

IMPORTANT:
- Extract UP TO 20 products maximum
- MUST find products if they exist - check JSON in scripts thoroughly
- If no products found after checking all sources, return empty array with low confidence
- Price range: ₹1 to ₹50000 (reject outliers)

JSON only:"""
    
    def _calculate_price_per_unit(self, price: float, quantity: Optional[Dict]) -> tuple[Optional[float], Optional[str]]:
        """Calculate price per unit from quantity. Returns (price_per_unit, display_string)"""
        if not quantity or not isinstance(quantity, dict):
            return None, None
        
        qty_value = quantity.get("value")
        qty_unit = (quantity.get("unit") or "").lower().strip()
        
        if not qty_value or qty_value <= 0:
            return None, None
        
        try:
            qty_value = float(qty_value)
        except (ValueError, TypeError):
            return None, None
        
        # Normalize to base units and calculate
        if qty_unit in ["g", "gm", "gram", "grams"]:
            # Weight: normalize to 100g
            base_value = qty_value  # already in grams
            per_unit = (price / base_value) * 100
            return per_unit, f"₹{per_unit:.2f}/100g"
        
        elif qty_unit in ["kg", "kilogram", "kilograms"]:
            # Weight: convert kg to g, normalize to 100g
            base_value = qty_value * 1000  # convert to grams
            per_unit = (price / base_value) * 100
            return per_unit, f"₹{per_unit:.2f}/100g"
        
        elif qty_unit in ["ml", "milliliter", "milliliters"]:
            # Volume: normalize to 100ml
            base_value = qty_value  # already in ml
            per_unit = (price / base_value) * 100
            return per_unit, f"₹{per_unit:.2f}/100ml"
        
        elif qty_unit in ["l", "liter", "litre", "liters", "litres"]:
            # Volume: convert L to ml, normalize to 100ml
            base_value = qty_value * 1000  # convert to ml
            per_unit = (price / base_value) * 100
            return per_unit, f"₹{per_unit:.2f}/100ml"
        
        elif qty_unit in ["pcs", "pc", "piece", "pieces", "count", "pack"]:
            # Count-based: price per piece
            per_unit = price / qty_value
            return per_unit, f"₹{per_unit:.2f}/pc"
        
        else:
            # Unknown unit, try to extract from name
            return None, None
    
    def _validate_products(
        self,
        products: List[Dict],
        platform: str,
        base_url: str
    ) -> List[Dict]:
        """Validate and enrich extracted products with price per unit calculation"""
        validated = []
        seen_names = set()
        
        for p in products:
            name = p.get("name", "").strip()
            price = p.get("price")

            # Extract quantity from AI response (preferred) or fallback to parsing name
            quantity = p.get("quantity")
            if not quantity or not isinstance(quantity, dict):
                # Fallback: try to extract from name
                quantity = self._extract_quantity_from_name(name)
            
            # Calculate price per unit
            price_per_unit, price_per_unit_display = self._calculate_price_per_unit(
                float(price) if price else 0, 
                quantity
            )
            
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
                "ai_extracted": True,
                "quantity": quantity,
                "price_per_unit": price_per_unit,
                "price_per_unit_display": price_per_unit_display
            })
        
        return validated[:20]  # Limit to 20 products
    
    def _extract_quantity_from_name(self, name: str) -> Optional[Dict]:
        """Fallback: Extract quantity from product name using regex"""
        if not name:
            return None
        
        # Patterns for weight/volume
        patterns = [
            (r"(\d+(?:\.\d+)?)\s*(kg|kilogram)", "kg", 1000),  # kg → convert to g
            (r"(\d+(?:\.\d+)?)\s*(g|gm|gram)", "g", 1),  # g
            (r"(\d+(?:\.\d+)?)\s*(l|liter|litre)", "ml", 1000),  # L → convert to ml
            (r"(\d+(?:\.\d+)?)\s*(ml|milliliter)", "ml", 1),  # ml
            (r"(\d+)\s*(pcs|pc|piece|pieces|count|pack)", "pcs", 1),  # count
        ]
        
        name_lower = name.lower()
        for pattern, unit, multiplier in patterns:
            match = re.search(pattern, name_lower)
            if match:
                value = float(match.group(1)) * multiplier
                if unit == "kg":
                    return {"value": value, "unit": "g"}
                elif unit == "ml" and multiplier == 1000:  # L converted
                    return {"value": value, "unit": "ml"}
                else:
                    return {"value": float(match.group(1)), "unit": unit}
        
        return None


# Singleton instance
_ai_scraper: Optional[AIScraper] = None


def get_ai_scraper() -> AIScraper:
    """Get or create the AI scraper singleton"""
    global _ai_scraper
    if _ai_scraper is None:
        _ai_scraper = AIScraper()
    return _ai_scraper
