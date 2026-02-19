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
from typing import List, Dict, Optional, Any, Set
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
    Uses Gemini 2.5 Flash for cost-effective, fast responses with model fallback.
    """
    
    # HARDCODED model priority: gemini-2.5-flash first, then lite, then gemma
    GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemma-3-27b-it"]
    
    def __init__(self, api_key: Optional[str] = None):
        self.model_name = self.GEMINI_MODELS[0]  # Primary model
        self.models = self.GEMINI_MODELS  # All models for fallback
        self.request_timeout_s = float(os.getenv("GEMINI_TIMEOUT_SEC", "60"))
        self.max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512"))
        self.max_input_products = int(os.getenv("GEMINI_MAX_INPUT_PRODUCTS", "120"))
        self.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
        self.top_p = float(os.getenv("GEMINI_TOP_P", "0.95"))
        self.use_http = os.getenv("GEMINI_USE_HTTP", "1").lower() in {"1", "true", "yes"}
        self.http_base_url = os.getenv(
            "GEMINI_HTTP_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta"
        )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        # Track unavailable/exhausted models for the day
        self._gemini_unavailable_models: Dict[str, str] = {}
        self._gemini_exhausted_models: Dict[str, str] = {}
        self._gemini_last_reset_date: str = ""
        
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
            f"fallbacks: {self.models[1:]}, "
            f"timeout: {self.request_timeout_s}s, "
            f"max_tokens: {self.max_output_tokens}, "
            f"transport: {transport})"
        )
    
    def is_available(self) -> bool:
        """Check if Gemini is properly configured"""
        return self.model is not None

    def _get_today(self) -> str:
        """Get today's date string for tracking"""
        import datetime
        return datetime.date.today().isoformat()
    
    def _reset_models_if_new_day(self):
        """Reset model tracking on new day"""
        today = self._get_today()
        if self._gemini_last_reset_date != today:
            self._gemini_unavailable_models = {}
            self._gemini_exhausted_models = {}
            self._gemini_last_reset_date = today
    
    def _is_model_available(self, model: str) -> bool:
        """Check if a specific Gemini model is available today"""
        self._reset_models_if_new_day()
        return model not in self._gemini_unavailable_models and model not in self._gemini_exhausted_models
    
    def _mark_model_unavailable(self, model: str):
        """Mark a Gemini model as unavailable (404/403 errors)"""
        self._gemini_unavailable_models[model] = self._get_today()
        print(f"⚠️ Gemini Service: Model {model} marked unavailable for today")
    
    def _mark_model_exhausted(self, model: str):
        """Mark a Gemini model as quota exhausted (429 errors)"""
        self._gemini_exhausted_models[model] = self._get_today()
        print(f"🔄 Gemini Service: Model {model} quota exhausted for today")
    
    def _get_available_models(self) -> List[str]:
        """Get list of models that are available today"""
        self._reset_models_if_new_day()
        return [m for m in self.models if self._is_model_available(m)]

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

    def _extract_relevant_names(self, text: str) -> List[str]:
        if not text:
            return []
        # Try to parse relevant_names array.
        match = re.search(r'"relevant_names"\s*:\s*\[(.*?)\]', text, re.S)
        if match:
            names = re.findall(r'"([^"]+)"', match.group(1))
            if names:
                return names
        # Fallback: grab "name" fields from relevant_products.
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', text)
        if names:
            return names
        # Fallback: bullet list lines.
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("-", "•")):
                lines.append(stripped.lstrip("-• ").strip())
        return [name for name in lines if name]

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
        id_map = {idx: prod for idx, prod in enumerate(original_products)}
        relevant: List[Dict[str, Any]] = []
        filtered: List[Dict[str, Any]] = []
        relevant_ids: Set[int] = set()

        def try_parse_id(value: Any) -> Optional[int]:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
            return None

        # Prefer id-based matching if AI returned ids.
        for item in ai_relevant:
            item_id = None
            item_dict: Dict[str, Any] = {}
            if isinstance(item, dict):
                item_dict = item
                item_id = try_parse_id(item.get("id") or item.get("idx"))
            else:
                item_id = try_parse_id(item)
            if item_id is None:
                continue
            orig = id_map.get(item_id)
            if not orig:
                continue
            enriched = {**orig}
            enriched["relevance_score"] = item_dict.get("relevance_score", 50)
            enriched["relevance_reason"] = item_dict.get("relevance_reason", "")
            relevant.append(enriched)
            relevant_ids.add(item_id)

        if relevant:
            for idx, orig in enumerate(original_products):
                if idx in relevant_ids:
                    continue
                filtered.append({
                    "name": orig.get("name"),
                    "platform": orig.get("platform"),
                    "filter_reason": "Filtered by AI"
                })
            # If AI provided explicit filtered ids, use those when available.
            explicit_filtered: List[Dict[str, Any]] = []
            for item in ai_filtered:
                item_id = None
                if isinstance(item, dict):
                    item_id = try_parse_id(item.get("id") or item.get("idx"))
                else:
                    item_id = try_parse_id(item)
                if item_id is None:
                    continue
                orig = id_map.get(item_id)
                if orig:
                    explicit_filtered.append({
                        "name": orig.get("name"),
                        "platform": orig.get("platform"),
                        "filter_reason": "Filtered by AI"
                    })
            if explicit_filtered:
                return relevant, explicit_filtered
            return relevant, filtered

        ai_items: List[Dict[str, Any]] = []
        for item in ai_relevant:
            if isinstance(item, dict):
                ai_items.append(item)
            else:
                ai_items.append({"name": str(item)})

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
            normalized_filtered: List[Dict[str, Any]] = []
            for item in ai_filtered:
                if isinstance(item, dict):
                    if "name" in item or "platform" in item or "filter_reason" in item:
                        normalized_filtered.append(item)
                    else:
                        normalized_filtered.append({
                            "name": str(item),
                            "filter_reason": "Filtered by AI"
                        })
                else:
                    normalized_filtered.append({
                        "name": str(item),
                        "filter_reason": "Filtered by AI"
                    })
            return relevant, normalized_filtered
        return relevant, filtered

    async def _generate_content(self, prompt: str) -> tuple[str, Dict[str, Optional[int]]]:
        if self.use_http:
            return await self._generate_content_http(prompt)
        response = await asyncio.to_thread(self.model.generate_content, prompt)
        return response.text, self._extract_usage_metadata(response)

    async def _generate_content_http(self, prompt: str) -> tuple[str, Dict[str, Optional[int]]]:
        """Generate content using HTTP API with model-level fallback"""
        models_to_try = self._get_available_models()
        
        if not models_to_try:
            raise Exception("All Gemini models exhausted or unavailable for today")
        
        last_error = None
        for model_name in models_to_try:
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
            
            try:
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
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                last_error = f"HTTP {status}"
                if status == 429:
                    self._mark_model_exhausted(model_name)
                    continue
                if status in {403, 404}:
                    self._mark_model_unavailable(model_name)
                    continue
                # Other errors, try next model
                print(f"⚠️ Gemini Service: Model {model_name} error: {last_error}, trying next model...")
                continue
            except Exception as e:
                last_error = str(e)
                print(f"⚠️ Gemini Service: Model {model_name} error: {last_error}, trying next model...")
                continue
        
        raise Exception(f"All Gemini models failed. Last error: {last_error}")

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
        text = ""
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
        
        # Limit products to avoid token limits (preserve input order)
        products_subset = products[:self.max_input_products]
        
        # Build the prompt
        prompt = self._build_filter_prompt(query, products_subset, strict_mode)
        
        text = ""
        start_time = time.monotonic()
        try:
            text, usage = await asyncio.wait_for(
                self._generate_content(prompt),
                timeout=self.request_timeout_s
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            try:
                result = self._parse_json_text(text)
                ai_relevant = result.get("relevant_items")
                if ai_relevant is None:
                    ai_relevant = result.get("relevant_products")
                if ai_relevant is None:
                    ai_relevant = result.get("relevant_names")
                if ai_relevant is None:
                    ai_relevant = []
                ai_filtered = result.get("filtered_ids")
                if ai_filtered is None:
                    ai_filtered = result.get("filtered_out")
                if ai_filtered is None:
                    ai_filtered = result.get("filtered_names")
                if ai_filtered is None:
                    ai_filtered = []
            except Exception:
                # Salvage AI output by extracting names from raw text.
                ai_relevant = self._extract_relevant_names(text)
                ai_filtered = []
                result = {
                    "query_understanding": {"original": query, "interpreted_as": query},
                    "relevant_names": ai_relevant,
                    "filtered_names": ai_filtered
                }

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
        
        text = ""
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
                "provider": "gemini",
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
                    "provider": "gemini",
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
                    "provider": "gemini",
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
        query_tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t]
        primary_hints = {
            "milk", "rice", "oil", "sugar", "salt", "tea", "coffee", "flour",
            "atta", "bread", "butter", "ghee", "cheese", "curd", "yogurt",
            "paneer", "egg", "eggs", "potato", "onion", "tomato", "apple",
            "banana", "orange", "chicken", "fish", "mutton", "dal", "lentil",
            "masala", "spice"
        }
        primary_product = next((t for t in query_tokens if t in primary_hints), None)
        if primary_product is None:
            primary_product = query_tokens[0] if query_tokens else query.lower().strip()
        optional_terms = [t for t in query_tokens if t != primary_product]

        products_json = json.dumps([
            {
                "id": i,
                "name": p.get("name", ""),
                "price": p.get("price", 0),
                "platform": p.get("platform", "")
            }
            for i, p in enumerate(products)
        ], separators=(",", ":"))

        strictness = "strict" if strict_mode else "lenient"

        return (
            f'Filter grocery products for query "{query}". '
            f"mode={strictness}. "
            f"Primary product term: {primary_product}. "
            f"Optional terms: {optional_terms}. "
            "If query has multiple words, keep items that match the primary product "
            "even if optional terms are missing, unless clearly a different product type. "
            "Return JSON with keys: "
            "query_understanding{original,interpreted_as,category,primary_product,optional_terms}, "
            "relevant_items[{id,relevance_score,relevance_reason}], "
            "filtered_ids[int]. "
            "Only use ids from the input list (no new items). "
            "Prefer keeping items when unsure. "
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
