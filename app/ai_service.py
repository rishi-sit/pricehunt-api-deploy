"""
AI Service for PriceHunt
Supports multiple AI providers: Mistral AI (default), Google Gemini (fallback)

Handles all interactions with AI for:
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
    """Product with relevance scoring from AI"""
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


class AIService:
    """
    Unified AI Service supporting multiple providers.
    Default: Mistral AI (1 billion tokens/month free)
    Fallback: Google Gemini
    """
    
    # Provider constants
    PROVIDER_MISTRAL = "mistral"
    PROVIDER_GEMINI = "gemini"
    
    def __init__(self, api_key: Optional[str] = None, provider: Optional[str] = None):
        # Determine provider (mistral by default)
        self.provider = (provider or os.getenv("AI_PROVIDER", "mistral")).lower()
        
        # Configuration
        self.request_timeout_s = float(os.getenv("AI_TIMEOUT_SEC", "60"))
        self.max_output_tokens = int(os.getenv("AI_MAX_OUTPUT_TOKENS", "512"))
        self.max_input_products = int(os.getenv("AI_MAX_INPUT_PRODUCTS", "120"))
        self.temperature = float(os.getenv("AI_TEMPERATURE", "0.1"))
        self.top_p = float(os.getenv("AI_TOP_P", "0.95"))
        
        # Provider-specific setup
        if self.provider == self.PROVIDER_MISTRAL:
            self._setup_mistral(api_key)
        else:
            self._setup_gemini(api_key)
    
    def _setup_mistral(self, api_key: Optional[str] = None):
        """Setup Mistral AI provider"""
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY") or os.getenv("AI_API_KEY")
        self.model_name = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
        self.base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
        
        if not self.api_key:
            print("⚠️ MISTRAL_API_KEY not set - AI features will be disabled")
            self._available = False
            return
        
        self._available = True
        print(
            f"✅ Mistral AI initialized (model: {self.model_name}, "
            f"timeout: {self.request_timeout_s}s, "
            f"max_tokens: {self.max_output_tokens})"
        )
    
    def _setup_gemini(self, api_key: Optional[str] = None):
        """Setup Google Gemini provider (fallback)"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.base_url = os.getenv(
            "GEMINI_HTTP_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta"
        )
        
        if not self.api_key:
            print("⚠️ GEMINI_API_KEY not set - AI features will be disabled")
            self._available = False
            return
        
        self._available = True
        print(
            f"✅ Gemini AI initialized (model: {self.model_name}, "
            f"timeout: {self.request_timeout_s}s, "
            f"max_tokens: {self.max_output_tokens})"
        )
    
    def is_available(self) -> bool:
        """Check if AI is properly configured"""
        return self._available
    
    async def _generate_content(self, prompt: str) -> tuple[str, Dict[str, Optional[int]]]:
        """Generate content using the configured provider"""
        if self.provider == self.PROVIDER_MISTRAL:
            return await self._generate_mistral(prompt)
        else:
            return await self._generate_gemini(prompt)
    
    async def _generate_mistral(self, prompt: str) -> tuple[str, Dict[str, Optional[int]]]:
        """Generate content using Mistral AI API (OpenAI-compatible)"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that analyzes grocery products. Always respond with valid JSON only, no explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"}
        }
        
        timeout = httpx.Timeout(
            timeout=self.request_timeout_s,
            connect=min(self.request_timeout_s, 10.0)
        )
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        
        # Extract text from response
        text = ""
        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            text = message.get("content", "")
        
        # Extract usage
        usage = data.get("usage", {})
        usage_info = {
            "prompt_token_count": usage.get("prompt_tokens"),
            "candidates_token_count": usage.get("completion_tokens"),
            "total_token_count": usage.get("total_tokens")
        }
        
        return text, usage_info
    
    async def _generate_gemini(self, prompt: str) -> tuple[str, Dict[str, Optional[int]]]:
        """Generate content using Google Gemini API"""
        model_name = self.model_name.strip()
        model_path = model_name if model_name.startswith("models/") else f"models/{model_name}"
        url = f"{self.base_url}/{model_path}:generateContent"
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
        
        # Extract text
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text = parts[0].get("text", "")
        
        # Extract usage
        usage = data.get("usageMetadata", {})
        usage_info = {
            "prompt_token_count": usage.get("promptTokenCount"),
            "candidates_token_count": usage.get("candidatesTokenCount"),
            "total_token_count": usage.get("totalTokenCount")
        }
        
        return text, usage_info
    
    def _parse_json_text(self, text: str) -> Dict[str, Any]:
        """Parse JSON from AI response text"""
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
            # Retry with trailing-comma cleanup
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            return json.loads(cleaned)
    
    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()
    
    def _extract_relevant_names(self, text: str) -> List[str]:
        """Extract product names from AI response text"""
        if not text:
            return []
        match = re.search(r'"relevant_names"\s*:\s*\[(.*?)\]', text, re.S)
        if match:
            names = re.findall(r'"([^"]+)"', match.group(1))
            if names:
                return names
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', text)
        if names:
            return names
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("-", "•")):
                lines.append(stripped.lstrip("-• ").strip())
        return [name for name in lines if name]
    
    def _match_ai_item(self, orig: Dict[str, Any], ai_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Match original product to AI response item"""
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
        """Apply AI filtering results to original products"""
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

        # Prefer id-based matching
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
            return relevant, filtered

        # Fallback to name-based matching
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

        return relevant, filtered
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Simple word overlap similarity"""
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    async def ping(self) -> Dict[str, Any]:
        """Lightweight AI connectivity test"""
        if not self.is_available():
            return {
                "ok": False,
                "error": "ai_unavailable",
                "provider": self.provider,
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
                "provider": self.provider,
                "model": self.model_name,
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
                "provider": self.provider,
                "model": self.model_name
            }
        except httpx.HTTPStatusError as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "ok": False,
                "timeout": False,
                "latency_ms": elapsed_ms,
                "provider": self.provider,
                "model": self.model_name,
                "status_code": e.response.status_code,
                "error": e.response.text[:200]
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "ok": False,
                "timeout": False,
                "latency_ms": elapsed_ms,
                "provider": self.provider,
                "model": self.model_name,
                "error": str(e)
            }
    
    async def filter_relevant_products(
        self, 
        query: str, 
        products: List[Dict[str, Any]],
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Use AI to filter products based on search query relevance.
        """
        if not self.is_available():
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "ai_meta": {
                    "ai_available": False,
                    "provider": self.provider,
                    "model": self.model_name
                }
            }
        
        products_subset = products[:self.max_input_products]
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
                ai_relevant = result.get("relevant_items") or result.get("relevant_products") or result.get("relevant_names") or []
                ai_filtered = result.get("filtered_ids") or result.get("filtered_out") or result.get("filtered_names") or []
            except Exception:
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
                "provider": self.provider,
                "model": self.model_name,
                "latency_ms": elapsed_ms,
                "token_usage": usage
            }
            
            return result
            
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ {self.provider} filter error: AI timeout")
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "error": "AI timeout",
                "ai_meta": {
                    "provider": self.provider,
                    "model": self.model_name,
                    "latency_ms": elapsed_ms,
                    "timeout": True
                }
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ {self.provider} filter error: {e}")
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "error": str(e),
                "ai_meta": {
                    "provider": self.provider,
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
        Use AI to match similar products across different platforms.
        """
        if not self.is_available():
            return {
                "product_groups": [],
                "unmatched_products": products,
                "ai_powered": False,
                "ai_meta": {
                    "ai_available": False,
                    "provider": self.provider,
                    "model": self.model_name
                }
            }
        
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
                "provider": self.provider,
                "model": self.model_name,
                "latency_ms": elapsed_ms,
                "token_usage": usage
            }
            
            # Enrich groups with full product data
            for group in result.get("product_groups", []):
                group["products"] = self._enrich_products(
                    group.get("products", []),
                    products
                )
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
            print(f"❌ {self.provider} matching error: AI timeout")
            return {
                "product_groups": [],
                "unmatched_products": products,
                "ai_powered": False,
                "error": "AI timeout",
                "ai_meta": {
                    "provider": self.provider,
                    "model": self.model_name,
                    "latency_ms": elapsed_ms,
                    "timeout": True
                }
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ {self.provider} matching error: {e}")
            return {
                "product_groups": [],
                "unmatched_products": products,
                "ai_powered": False,
                "error": str(e),
                "ai_meta": {
                    "provider": self.provider,
                    "model": self.model_name,
                    "latency_ms": elapsed_ms,
                    "timeout": False,
                    "response_preview": (text or "")[:200]
                }
            }
    
    async def understand_query(self, query: str) -> Dict[str, Any]:
        """
        Use AI to understand the user's search intent.
        """
        if not self.is_available():
            return {
                "original_query": query,
                "product_type": query,
                "ai_powered": False,
                "ai_meta": {
                    "ai_available": False,
                    "provider": self.provider,
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
                "provider": self.provider,
                "model": self.model_name,
                "latency_ms": elapsed_ms,
                "token_usage": usage
            }
            return result

        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ {self.provider} query understanding error: AI timeout")
            return {
                "original_query": query,
                "product_type": query,
                "ai_powered": False,
                "error": "AI timeout",
                "ai_meta": {
                    "provider": self.provider,
                    "model": self.model_name,
                    "latency_ms": elapsed_ms,
                    "timeout": True
                }
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ {self.provider} query understanding error: {e}")
            return {
                "original_query": query,
                "product_type": query,
                "ai_powered": False,
                "error": str(e),
                "ai_meta": {
                    "provider": self.provider,
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
            name = str(ai_prod.get("name", "")).lower()
            platform = str(ai_prod.get("platform", "")).lower()
            if not name:
                enriched.append(ai_prod)
                continue
            
            for orig in original_products:
                orig_name = str(orig.get("name", "")).lower()
                orig_platform = str(orig.get("platform", "")).lower()
                
                platform_matches = not platform or platform in orig_platform or orig_platform in platform
                if platform_matches:
                    if name in orig_name or orig_name in name or \
                       self._name_similarity(name, orig_name) > 0.8:
                        enriched_product = {**orig}
                        enriched_product["relevance_score"] = ai_prod.get("relevance_score", 50)
                        enriched_product["relevance_reason"] = ai_prod.get("relevance_reason", "")
                        enriched.append(enriched_product)
                        break
            else:
                enriched.append(ai_prod)
        
        return enriched


# Singleton instance
_ai_service: Optional[AIService] = None


def get_ai_service() -> AIService:
    """Get or create the AI service singleton"""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService()
    return _ai_service


# Backward compatibility alias
def get_gemini_service() -> AIService:
    """Alias for backward compatibility with existing code"""
    return get_ai_service()


# Also export as GeminiService for full backward compatibility
GeminiService = AIService
