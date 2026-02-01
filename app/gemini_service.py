"""
Gemini AI Service for PriceHunt
Handles all interactions with Google's Gemini API for:
1. Smart search filtering
2. Product matching across platforms
3. Natural language query understanding
"""
import os
import json
import asyncio
import time
import re
from typing import List, Dict, Optional, Any
import google.generativeai as genai
import httpx
from pydantic import BaseModel


class ProductInput(BaseModel):
    """Product data from Android scraping"""
    name: str
    price: float
    original_price: Optional[float] = None
    platform: str
    url: Optional[str] = None
    image_url: Optional[str] = None
    available: bool = True


class RelevantProduct(BaseModel):
    """Product with relevance scoring from Gemini"""
    name: str
    price: float
    platform: str
    relevance_score: int  # 0-100
    relevance_reason: str
    original_price: Optional[float] = None
    url: Optional[str] = None
    image_url: Optional[str] = None


class FilteredProduct(BaseModel):
    """Product that was filtered out"""
    name: str
    platform: str
    filter_reason: str


class ProductGroup(BaseModel):
    """Group of matched products across platforms"""
    canonical_name: str
    brand: Optional[str] = None
    quantity: Optional[str] = None
    products: List[Dict[str, Any]]
    best_deal: Dict[str, Any]
    price_range: str


class GeminiService:
    """
    Service for Gemini AI interactions.
    Uses Gemini 1.5 Flash for cost-effective, fast responses.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.request_timeout_s = float(os.getenv("GEMINI_TIMEOUT_SEC", "60"))
        self.max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512"))
        self.max_input_products = int(os.getenv("GEMINI_MAX_INPUT_PRODUCTS", "30"))
        self.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
        self.top_p = float(os.getenv("GEMINI_TOP_P", "0.95"))
        self.use_http = os.getenv("GEMINI_USE_HTTP", "1").lower() in {"1", "true", "yes"}
        self.http_base_url = os.getenv(
            "GEMINI_HTTP_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta"
        )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            print("⚠️ GEMINI_API_KEY not set - AI features will be disabled")
            self.model = None
            return
            
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 2.5 Flash for speed and cost efficiency
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,  # Low temperature for consistency
                "top_p": self.top_p,
                "max_output_tokens": self.max_output_tokens,
                "response_mime_type": "application/json"  # Force JSON output
            }
        )
        transport = "http" if self.use_http else "sdk"
        print(
            f"✅ Gemini AI initialized (model: {self.model_name}, "
            f"timeout: {self.request_timeout_s}s, "
            f"max_tokens: {self.max_output_tokens}, "
            f"transport: {transport})"
        )
    
    def is_available(self) -> bool:
        """Check if Gemini is properly configured"""
        return self.model is not None

    def _extract_usage_metadata(self, response: Any) -> Dict[str, Optional[int]]:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return {}
        return {
            "prompt_token_count": getattr(usage, "prompt_token_count", None),
            "candidates_token_count": getattr(usage, "candidates_token_count", None),
            "total_token_count": getattr(usage, "total_token_count", None)
        }

    def _extract_http_usage(self, data: Dict[str, Any]) -> Dict[str, Optional[int]]:
        usage = data.get("usageMetadata", {}) if isinstance(data, dict) else {}
        return {
            "prompt_token_count": usage.get("promptTokenCount"),
            "candidates_token_count": usage.get("candidatesTokenCount"),
            "total_token_count": usage.get("totalTokenCount")
        }

    def _parse_json_text(self, text: str) -> Dict[str, Any]:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("empty_response")
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Retry with a simple trailing-comma cleanup.
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            return json.loads(cleaned)

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()

    def _match_ai_item(self, orig: Dict[str, Any], ai_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        orig_name = self._normalize_text(orig.get("name", ""))
        if not orig_name:
            return None
        orig_platform = self._normalize_text(orig.get("platform", ""))
        best_item = None
        best_score = 0.0
        for item in ai_items:
            item_name = self._normalize_text(item.get("name", ""))
            if not item_name:
                continue
            item_platform = self._normalize_text(item.get("platform", ""))
            if item_platform and orig_platform and item_platform not in orig_platform and orig_platform not in item_platform:
                continue
            if item_name in orig_name or orig_name in item_name:
                score = 1.0
            else:
                score = self._name_similarity(item_name, orig_name)
            if score > best_score:
                best_score = score
                best_item = item
        return best_item if best_score >= 0.6 else None

    def _apply_ai_filter(
        self,
        ai_relevant: List[Any],
        ai_filtered: List[Any],
        original_products: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        ai_items: List[Dict[str, Any]] = []
        for item in ai_relevant:
            if isinstance(item, dict):
                ai_items.append(item)
            else:
                ai_items.append({"name": str(item)})

        relevant: List[Dict[str, Any]] = []
        filtered: List[Dict[str, Any]] = []
        for orig in original_products:
            match = self._match_ai_item(orig, ai_items)
            if match:
                enriched = {**orig}
                enriched["relevance_score"] = match.get("relevance_score", 50)
                enriched["relevance_reason"] = match.get("relevance_reason", "")
                relevant.append(enriched)
            else:
                filtered.append({
                    "name": orig.get("name"),
                    "platform": orig.get("platform"),
                    "filter_reason": "Filtered by AI"
                })

        if ai_filtered:
            return relevant, ai_filtered
        return relevant, filtered

    async def _generate_content(self, prompt: str) -> tuple[str, Dict[str, Optional[int]]]:
        if self.use_http:
            return await self._generate_content_http(prompt)
        response = await asyncio.to_thread(self.model.generate_content, prompt)
        return response.text, self._extract_usage_metadata(response)

    async def _generate_content_http(self, prompt: str) -> tuple[str, Dict[str, Optional[int]]]:
        model_name = self.model_name.strip()
        model_path = model_name if model_name.startswith("models/") else f"models/{model_name}"
        url = f"{self.http_base_url}/{model_path}:generateContent"
        params = {"key": self.api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json"
            }
        }
        timeout = httpx.Timeout(
            timeout=self.request_timeout_s,
            connect=min(self.request_timeout_s, 10.0)
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, params=params, json=payload)
            response.raise_for_status()
            data = response.json()

        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text = parts[0].get("text", "")

        return text, self._extract_http_usage(data)

    async def ping(self) -> Dict[str, Any]:
        """
        Lightweight Gemini connectivity test.
        Returns latency, transport, and any error details.
        """
        if not self.is_available():
            return {
                "ok": False,
                "error": "ai_unavailable",
                "model": self.model_name
            }
        prompt = 'Return JSON: {"ok": true}'
        start_time = time.monotonic()
        try:
            text, usage = await asyncio.wait_for(
                self._generate_content(prompt),
                timeout=self.request_timeout_s
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            parsed = None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            return {
                "ok": True,
                "latency_ms": elapsed_ms,
                "model": self.model_name,
                "transport": "http" if self.use_http else "sdk",
                "token_usage": usage,
                "parsed_ok": parsed,
                "response_preview": text[:200]
            }
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "ok": False,
                "timeout": True,
                "latency_ms": elapsed_ms,
                "model": self.model_name,
                "transport": "http" if self.use_http else "sdk"
            }
        except httpx.HTTPStatusError as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "ok": False,
                "timeout": False,
                "latency_ms": elapsed_ms,
                "model": self.model_name,
                "transport": "http",
                "status_code": e.response.status_code,
                "error": e.response.text[:200]
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "ok": False,
                "timeout": False,
                "latency_ms": elapsed_ms,
                "model": self.model_name,
                "transport": "http" if self.use_http else "sdk",
                "error": str(e)
            }

    async def list_models(self) -> Dict[str, Any]:
        """List available Gemini models for this API key."""
        if not self.is_available():
            return {
                "ok": False,
                "error": "ai_unavailable"
            }
        if not self.use_http:
            return {
                "ok": False,
                "error": "list_models_http_only"
            }
        url = f"{self.http_base_url}/models"
        params = {"key": self.api_key}
        timeout = httpx.Timeout(
            timeout=self.request_timeout_s,
            connect=min(self.request_timeout_s, 10.0)
        )
        start_time = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            models = data.get("models", [])
            simplified = [
                {
                    "name": m.get("name"),
                    "displayName": m.get("displayName"),
                    "supportedGenerationMethods": m.get("supportedGenerationMethods"),
                    "inputTokenLimit": m.get("inputTokenLimit"),
                    "outputTokenLimit": m.get("outputTokenLimit")
                }
                for m in models
            ]
            return {
                "ok": True,
                "latency_ms": elapsed_ms,
                "model_count": len(models),
                "models": simplified
            }
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "ok": False,
                "timeout": True,
                "latency_ms": elapsed_ms
            }
        except httpx.HTTPStatusError as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "ok": False,
                "timeout": False,
                "latency_ms": elapsed_ms,
                "status_code": e.response.status_code,
                "error": e.response.text[:500]
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "ok": False,
                "timeout": False,
                "latency_ms": elapsed_ms,
                "error": str(e)
            }
    
    async def filter_relevant_products(
        self, 
        query: str, 
        products: List[Dict[str, Any]],
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Use Gemini to filter products based on search query relevance.
        
        Args:
            query: User's search query (e.g., "milk")
            products: List of scraped products from all platforms
            strict_mode: If True, be strict about filtering (recommended for single words)
            
        Returns:
            {
                "relevant_products": [...],
                "filtered_out": [...],
                "query_understanding": {...}
            }
        """
        if not self.is_available():
            # Fallback: return all products if Gemini unavailable
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "ai_meta": {
                    "ai_available": False,
                    "model": self.model_name
                }
            }
        
        # Limit products to avoid token limits (prioritize by price)
        sorted_products = sorted(products, key=lambda x: x.get("price", float("inf")))
        products_subset = sorted_products[:self.max_input_products]
        
        # Build the prompt
        prompt = self._build_filter_prompt(query, products_subset, strict_mode)
        
        start_time = time.monotonic()
        try:
            text, usage = await asyncio.wait_for(
                self._generate_content(prompt),
                timeout=self.request_timeout_s
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            result = self._parse_json_text(text)
            ai_relevant = result.get("relevant_products", [])
            ai_filtered = result.get("filtered_out", [])
            relevant, filtered = self._apply_ai_filter(ai_relevant, ai_filtered, products)
            result["relevant_products"] = relevant
            result["filtered_out"] = filtered
            result["ai_powered"] = True
            result["ai_meta"] = {
                "model": self.model_name,
                "latency_ms": elapsed_ms,
                "token_usage": usage,
                "transport": "http" if self.use_http else "sdk"
            }
            
            return result
            
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print("❌ Gemini filter error: AI timeout")
            # Fallback on error
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "error": "AI timeout",
                "ai_meta": {
                    "model": self.model_name,
                    "latency_ms": elapsed_ms,
                    "timeout": True
                }
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ Gemini filter error: {e}")
            # Fallback on error
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "error": str(e),
                "ai_meta": {
                    "model": self.model_name,
                    "latency_ms": elapsed_ms,
                    "timeout": False,
                    "response_preview": (text or "")[:200]
                }
            }
    
    async def match_products_across_platforms(
        self, 
        products: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use Gemini to match similar products across different platforms.
        
        Args:
            products: List of products from multiple platforms
            
        Returns:
            {
                "product_groups": [
                    {
                        "canonical_name": "Amul Toned Milk 500ml",
                        "products": [...],
                        "best_deal": {...}
                    }
                ],
                "unmatched_products": [...]
            }
        """
        if not self.is_available():
            # Fallback: no grouping
            return {
                "product_groups": [],
                "unmatched_products": products,
                "ai_powered": False,
                "ai_meta": {
                    "ai_available": False,
                    "model": self.model_name
                }
            }
        
        # Limit products
        products_subset = products[:self.max_input_products]
        
        prompt = self._build_matching_prompt(products_subset)
        
        start_time = time.monotonic()
        try:
            text, usage = await asyncio.wait_for(
                self._generate_content(prompt),
                timeout=self.request_timeout_s
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            result = self._parse_json_text(text)
            result["ai_powered"] = True
            result["ai_meta"] = {
                "model": self.model_name,
                "latency_ms": elapsed_ms,
                "token_usage": usage,
                "transport": "http" if self.use_http else "sdk"
            }
            
            # Enrich groups with full product data
            for group in result.get("product_groups", []):
                group["products"] = self._enrich_products(
                    group.get("products", []),
                    products
                )
                # Recalculate best deal with full data
                if group["products"]:
                    best = min(group["products"], key=lambda x: x.get("price", float("inf")))
                    group["best_deal"] = {
                        "platform": best.get("platform"),
                        "price": best.get("price"),
                        "name": best.get("name")
                    }
            
            return result
            
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print("❌ Gemini matching error: AI timeout")
            return {
                "product_groups": [],
                "unmatched_products": products,
                "ai_powered": False,
                "error": "AI timeout",
                "ai_meta": {
                    "model": self.model_name,
                    "latency_ms": elapsed_ms,
                    "timeout": True
                }
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ Gemini matching error: {e}")
            return {
                "product_groups": [],
                "unmatched_products": products,
                "ai_powered": False,
                "error": str(e),
                "ai_meta": {
                    "model": self.model_name,
                    "latency_ms": elapsed_ms,
                    "timeout": False,
                    "response_preview": (text or "")[:200]
                }
            }
    
    async def understand_query(self, query: str) -> Dict[str, Any]:
        """
        Use Gemini to understand the user's search intent.
        
        Returns:
            {
                "original_query": "milk 500ml",
                "product_type": "milk",
                "quantity": "500ml",
                "brand": null,
                "category": "dairy",
                "search_terms": ["milk", "toned milk", "full cream milk"],
                "exclude_terms": ["milkshake", "chocolate milk", "milk powder"]
            }
        """
        if not self.is_available():
            return {
                "original_query": query,
                "product_type": query,
                "ai_powered": False,
                "ai_meta": {
                    "ai_available": False,
                    "model": self.model_name
                }
            }
        
        prompt = f"""Analyze this grocery search query and extract structured information.

Query: "{query}"

Return a JSON object with:
- original_query: the exact query
- product_type: the main product being searched (e.g., "milk", "rice", "oil")
- quantity: any quantity mentioned (e.g., "500ml", "1kg") or null
- brand: any brand mentioned or null
- category: product category (dairy, grocery, snacks, beverages, fruits, vegetables, meat, etc.)
- is_specific: true if user wants a specific product, false if general search
- search_terms: list of related terms to include in search
- exclude_terms: list of terms that should NOT be in results (derivatives, different products)

Be smart about exclude_terms:
- For "milk": exclude milkshake, chocolate milk, milk powder, condensed milk (unless specifically asked)
- For "apple": exclude apple juice, apple cider, apple pie
- For "oil": DON'T exclude anything - user wants oil products

JSON only, no explanation:"""
        
        start_time = time.monotonic()
        try:
            text, usage = await asyncio.wait_for(
                self._generate_content(prompt),
                timeout=self.request_timeout_s
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            result = self._parse_json_text(text)
            result["ai_powered"] = True
            result["ai_meta"] = {
                "model": self.model_name,
                "latency_ms": elapsed_ms,
                "token_usage": usage,
                "transport": "http" if self.use_http else "sdk"
            }
            return result

        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print("❌ Gemini query understanding error: AI timeout")
            return {
                "original_query": query,
                "product_type": query,
                "ai_powered": False,
                "error": "AI timeout",
                "ai_meta": {
                    "model": self.model_name,
                    "latency_ms": elapsed_ms,
                    "timeout": True
                }
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ Gemini query understanding error: {e}")
            return {
                "original_query": query,
                "product_type": query,
                "ai_powered": False,
                "error": str(e),
                "ai_meta": {
                    "model": self.model_name,
                    "latency_ms": elapsed_ms,
                    "timeout": False,
                    "response_preview": (text or "")[:200]
                }
            }
    
    def _build_filter_prompt(
        self, 
        query: str, 
        products: List[Dict], 
        strict_mode: bool
    ) -> str:
        """Build the prompt for product filtering"""
        
        products_json = json.dumps([
            {"name": p.get("name", ""), "price": p.get("price", 0), "platform": p.get("platform", "")}
            for p in products
        ], separators=(",", ":"))
        
        strictness = "strict" if strict_mode else "lenient"
        
        return (
            f'Filter grocery products for query "{query}". '
            f"mode={strictness}. "
            "Return JSON with keys: "
            "query_understanding{original,interpreted_as,category}, "
            "relevant_products[{name,price,platform,relevance_score,relevance_reason}], "
            "filtered_out[{name,platform,filter_reason}]. "
            f"Input products: {products_json}"
        )
    
    def _build_matching_prompt(self, products: List[Dict]) -> str:
        """Build the prompt for product matching"""
        
        products_json = json.dumps([
            {
                "name": p.get("name", ""),
                "price": p.get("price", 0),
                "platform": p.get("platform", "")
            }
            for p in products
        ], separators=(",", ":"))
        
        return (
            "Group identical products across platforms (same brand, size, variant). "
            "Return JSON with keys: "
            "product_groups[{canonical_name,brand,quantity,products[{name,price,platform}],"
            "best_deal{platform,price},price_range}], "
            "unmatched_products[{name,price,platform}]. "
            "Only groups with 2+ products from different platforms. "
            f"Input products: {products_json}"
        )
    
    def _enrich_products(
        self, 
        ai_products: List[Dict], 
        original_products: List[Dict]
    ) -> List[Dict]:
        """Enrich AI-returned products with full original data"""
        enriched = []
        
        for ai_prod in ai_products:
            if not isinstance(ai_prod, dict):
                ai_prod = {"name": str(ai_prod)}
            # Find matching original product
            name = str(ai_prod.get("name", "")).lower()
            platform = str(ai_prod.get("platform", "")).lower()
            if not name:
                enriched.append(ai_prod)
                continue
            
            for orig in original_products:
                orig_name = str(orig.get("name", "")).lower()
                orig_platform = str(orig.get("platform", "")).lower()
                
                # Match by name similarity and platform
                platform_matches = not platform or platform in orig_platform or orig_platform in platform
                if platform_matches:
                    if name in orig_name or orig_name in name or \
                       self._name_similarity(name, orig_name) > 0.8:
                        # Merge AI insights with original data
                        enriched_product = {**orig}
                        enriched_product["relevance_score"] = ai_prod.get("relevance_score", 50)
                        enriched_product["relevance_reason"] = ai_prod.get("relevance_reason", "")
                        enriched.append(enriched_product)
                        break
            else:
                # No match found, use AI product as-is
                enriched.append(ai_prod)
        
        return enriched
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Simple word overlap similarity"""
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)


# Singleton instance
_gemini_service: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """Get or create the Gemini service singleton"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service
