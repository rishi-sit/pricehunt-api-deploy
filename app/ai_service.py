"""
AI Service for PriceHunt
Supports multiple AI providers with smart quota management:
- Primary: Groq (fastest, 6000 req/day free)
- Fallback: Mistral AI (1B tokens/month free)
- Fallback: Google Gemini

Smart Quota Management:
- Tracks daily usage for each provider
- Auto-switches when quota exceeded (HTTP 429)
- Resets back to primary (Groq) at midnight UTC
- Maximizes free tier usage across all providers

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
from datetime import datetime, timezone
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


class QuotaTracker:
    """
    Tracks API quota usage per provider with daily reset.
    When a provider hits quota limit (429), marks it as exhausted for the day.
    Resets all providers back to available at midnight UTC.
    """
    
    def __init__(self):
        self._exhausted_providers: Dict[str, str] = {}  # provider -> date_exhausted
        self._request_counts: Dict[str, int] = {}  # provider -> count today
        self._last_reset_date: str = self._get_today()
    
    def _get_today(self) -> str:
        """Get today's date in UTC as string"""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    def _check_daily_reset(self):
        """Reset quota tracking if it's a new day"""
        today = self._get_today()
        if today != self._last_reset_date:
            print(f"🔄 New day ({today}) - resetting all provider quotas")
            self._exhausted_providers.clear()
            self._request_counts.clear()
            self._last_reset_date = today
    
    def mark_exhausted(self, provider: str):
        """Mark a provider as quota-exhausted for today"""
        self._check_daily_reset()
        today = self._get_today()
        self._exhausted_providers[provider] = today
        count = self._request_counts.get(provider, 0)
        print(f"⚠️ {provider} quota exhausted for {today} (after {count} requests)")
    
    def is_available(self, provider: str) -> bool:
        """Check if provider is available (not quota-exhausted today)"""
        self._check_daily_reset()
        return provider not in self._exhausted_providers
    
    def record_request(self, provider: str):
        """Record a successful request to a provider"""
        self._check_daily_reset()
        self._request_counts[provider] = self._request_counts.get(provider, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current quota status"""
        self._check_daily_reset()
        return {
            "date": self._last_reset_date,
            "request_counts": dict(self._request_counts),
            "exhausted_providers": list(self._exhausted_providers.keys()),
            "available_providers": [p for p in ["groq", "mistral", "gemini"] 
                                   if p not in self._exhausted_providers]
        }


# Global quota tracker (shared across requests)
_quota_tracker = QuotaTracker()


class AIService:
    """
    Unified AI Service supporting multiple providers with smart quota management.
    
    Priority order:
    1. Groq (fastest - uses custom LPU hardware)
    2. Mistral AI (1B tokens/month free)
    3. Google Gemini (fallback)
    
    Smart quota handling:
    - If provider returns 429 (quota exceeded), marks it exhausted for the day
    - Automatically falls back to next available provider
    - At midnight UTC, resets and tries Groq again
    """
    
    # Provider constants
    PROVIDER_GROQ = "groq"
    PROVIDER_MISTRAL = "mistral"
    PROVIDER_GEMINI = "gemini"
    
    # Rate limit error codes that trigger fallback
    RATE_LIMIT_CODES = {429, 503}
    
    def __init__(self):
        # Configuration
        self.request_timeout_s = float(os.getenv("AI_TIMEOUT_SEC", "30"))
        self.max_output_tokens = int(os.getenv("AI_MAX_OUTPUT_TOKENS", "512"))
        self.max_input_products = int(os.getenv("AI_MAX_INPUT_PRODUCTS", "120"))
        self.temperature = float(os.getenv("AI_TEMPERATURE", "0.1"))
        self.top_p = float(os.getenv("AI_TOP_P", "0.95"))
        
        # Setup all available providers
        self.providers: Dict[str, Dict[str, Any]] = {}
        self._setup_groq()
        self._setup_mistral()
        self._setup_gemini()
        
        # Reference to global quota tracker
        self.quota = _quota_tracker
        
        # Check availability
        self._available = len(self.providers) > 0
        
        if self._available:
            print(f"✅ AI Service initialized")
            for name, config in self.providers.items():
                print(f"   ✓ {name}: {config['model']}")
        else:
            print("⚠️ No AI providers configured - AI features disabled")
    
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
        # Priority order: Groq (testing) > Gemini > Mistral
        priority = [self.PROVIDER_GROQ, self.PROVIDER_GEMINI, self.PROVIDER_MISTRAL]
        available = []
        for p in priority:
            if p in self.providers and self.quota.is_available(p):
                available.append(p)
        return available
    
    def _get_primary_provider(self) -> Optional[str]:
        """Get the best available provider right now"""
        available = self._get_available_providers()
        return available[0] if available else None
    
    @property
    def provider(self) -> Optional[str]:
        """Current primary provider (dynamic based on quota)"""
        return self._get_primary_provider()
    
    @property
    def model_name(self) -> str:
        """Get current provider's model name"""
        provider = self.provider
        if provider and provider in self.providers:
            return self.providers[provider]["model"]
        return "unknown"
    
    def is_available(self) -> bool:
        """Check if any AI provider is available"""
        return self._available and len(self._get_available_providers()) > 0
    
    def get_quota_stats(self) -> Dict[str, Any]:
        """Get current quota usage statistics"""
        stats = self.quota.get_stats()
        stats["current_provider"] = self.provider
        stats["configured_providers"] = list(self.providers.keys())
        return stats
    
    async def _generate_content(
        self, 
        prompt: str, 
        provider: Optional[str] = None
    ) -> tuple[str, Dict[str, Optional[int]], str]:
        """
        Generate content using specified provider.
        Returns: (text, usage_info, provider_used)
        """
        provider = provider or self.provider
        if not provider or provider not in self.providers:
            raise ValueError("No AI provider available")
        
        config = self.providers[provider]
        
        if config["type"] == "openai_compatible":
            text, usage = await self._generate_openai_compatible(prompt, config)
        else:
            text, usage = await self._generate_gemini(prompt, config)
        
        # Record successful request
        self.quota.record_request(provider)
        
        return text, usage, provider
    
    async def _generate_openai_compatible(
        self, 
        prompt: str, 
        config: Dict[str, Any]
    ) -> tuple[str, Dict[str, Optional[int]]]:
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
                    "content": "You are a helpful assistant that analyzes grocery products. Always respond with valid JSON only, no markdown, no explanation."
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
        # Mistral supports it, Groq's mixtral may not
        if config.get("supports_json_mode", False):
            payload["response_format"] = {"type": "json_object"}
        
        timeout = httpx.Timeout(
            timeout=self.request_timeout_s,
            connect=min(self.request_timeout_s, 10.0)
        )
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                error_text = response.text[:500] if response.text else "No error body"
                print(f"❌ API Error {response.status_code} from {config.get('model')}: {error_text}")
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
    
    async def _generate_gemini(
        self, 
        prompt: str, 
        config: Dict[str, Any]
    ) -> tuple[str, Dict[str, Optional[int]]]:
        """Generate content using Google Gemini API"""
        model_name = config["model"].strip()
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
    
    async def _generate_with_fallback(
        self, 
        prompt: str
    ) -> tuple[str, Dict[str, Optional[int]], str, Optional[str]]:
        """
        Generate content with automatic fallback on quota exhaustion.
        
        When a provider returns 429 (quota exceeded):
        1. Marks that provider as exhausted for today
        2. Tries the next available provider
        3. Tomorrow, will try the exhausted provider again
        
        Returns: (text, usage_info, provider_used, fallback_reason)
        """
        providers_to_try = self._get_available_providers()
        
        if not providers_to_try:
            raise Exception("All AI providers exhausted for today")
        
        last_error = None
        fallback_reason = None
        original_provider = providers_to_try[0]
        
        for i, provider in enumerate(providers_to_try):
            try:
                text, usage, used_provider = await self._generate_content(prompt, provider)
                if i > 0:
                    fallback_reason = f"Switched from {original_provider}: {last_error}"
                return text, usage, used_provider, fallback_reason
                
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}"
                
                if e.response.status_code == 429:
                    # Quota exceeded - mark as exhausted for today
                    self.quota.mark_exhausted(provider)
                    print(f"🔄 {provider} quota exceeded, switching to fallback...")
                    continue
                elif e.response.status_code in self.RATE_LIMIT_CODES:
                    print(f"⚠️ {provider} rate limited (HTTP {e.response.status_code}), trying fallback...")
                    continue
                else:
                    print(f"⚠️ {provider} error: {last_error}, trying fallback...")
                    continue
                
            except asyncio.TimeoutError:
                last_error = "timeout"
                print(f"⚠️ {provider} timeout, trying fallback...")
                continue
                
            except Exception as e:
                last_error = str(e)
                print(f"⚠️ {provider} error: {last_error}, trying fallback...")
                continue
        
        # All providers failed
        raise Exception(f"All AI providers failed. Last error: {last_error}")
    
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
    
    def _normalize_relevance_score(self, raw_score: Any) -> int:
        """Normalize relevance score to integer 0-100"""
        if raw_score is None:
            return 50
        if isinstance(raw_score, float) and raw_score <= 1.0:
            return int(raw_score * 100)
        try:
            return int(raw_score)
        except (ValueError, TypeError):
            return 50
    
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
            enriched["relevance_score"] = self._normalize_relevance_score(item_dict.get("relevance_score", 50))
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
                enriched["relevance_score"] = self._normalize_relevance_score(match.get("relevance_score", 50))
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
                "quota_stats": self.get_quota_stats()
            }
        
        prompt = 'Return JSON: {"ok": true}'
        start_time = time.monotonic()
        try:
            text, usage, provider_used, fallback_reason = await asyncio.wait_for(
                self._generate_with_fallback(prompt),
                timeout=self.request_timeout_s
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            parsed = None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            
            result = {
                "ok": True,
                "latency_ms": elapsed_ms,
                "provider": provider_used,
                "model": self.providers[provider_used]["model"],
                "token_usage": usage,
                "parsed_ok": parsed,
                "response_preview": text[:200],
                "providers_configured": list(self.providers.keys()),
                "quota_stats": self.get_quota_stats()
            }
            if fallback_reason:
                result["fallback_reason"] = fallback_reason
            return result
            
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "ok": False,
                "timeout": True,
                "latency_ms": elapsed_ms,
                "quota_stats": self.get_quota_stats()
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "ok": False,
                "timeout": False,
                "latency_ms": elapsed_ms,
                "error": str(e),
                "quota_stats": self.get_quota_stats()
            }
    
    async def filter_relevant_products(
        self, 
        query: str, 
        products: List[Dict[str, Any]],
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Use AI to filter products based on search query relevance.
        Automatically falls back to next provider if quota exceeded.
        """
        if not self.is_available():
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "ai_meta": {
                    "ai_available": False,
                    "quota_stats": self.get_quota_stats()
                }
            }
        
        products_subset = products[:self.max_input_products]
        prompt = self._build_filter_prompt(query, products_subset, strict_mode)
        
        text = ""
        start_time = time.monotonic()
        try:
            text, usage, provider_used, fallback_reason = await asyncio.wait_for(
                self._generate_with_fallback(prompt),
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
                "provider": provider_used,
                "model": self.providers[provider_used]["model"],
                "latency_ms": elapsed_ms,
                "token_usage": usage,
                "quota_stats": self.get_quota_stats()
            }
            if fallback_reason:
                result["ai_meta"]["fallback_reason"] = fallback_reason
            
            return result
            
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ AI filter error: All providers timed out")
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "error": "AI timeout",
                "ai_meta": {
                    "latency_ms": elapsed_ms,
                    "timeout": True,
                    "quota_stats": self.get_quota_stats()
                }
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ AI filter error: {e}")
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "error": str(e),
                "ai_meta": {
                    "latency_ms": elapsed_ms,
                    "timeout": False,
                    "response_preview": (text or "")[:200],
                    "quota_stats": self.get_quota_stats()
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
                    "quota_stats": self.get_quota_stats()
                }
            }
        
        products_subset = products[:self.max_input_products]
        prompt = self._build_matching_prompt(products_subset)
        
        text = ""
        start_time = time.monotonic()
        try:
            text, usage, provider_used, fallback_reason = await asyncio.wait_for(
                self._generate_with_fallback(prompt),
                timeout=self.request_timeout_s
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            result = self._parse_json_text(text)
            result["ai_powered"] = True
            result["ai_meta"] = {
                "provider": provider_used,
                "model": self.providers[provider_used]["model"],
                "latency_ms": elapsed_ms,
                "token_usage": usage,
                "quota_stats": self.get_quota_stats()
            }
            if fallback_reason:
                result["ai_meta"]["fallback_reason"] = fallback_reason
            
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
            print(f"❌ AI matching error: All providers timed out")
            return {
                "product_groups": [],
                "unmatched_products": products,
                "ai_powered": False,
                "error": "AI timeout",
                "ai_meta": {
                    "latency_ms": elapsed_ms,
                    "timeout": True,
                    "quota_stats": self.get_quota_stats()
                }
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ AI matching error: {e}")
            return {
                "product_groups": [],
                "unmatched_products": products,
                "ai_powered": False,
                "error": str(e),
                "ai_meta": {
                    "latency_ms": elapsed_ms,
                    "timeout": False,
                    "response_preview": (text or "")[:200],
                    "quota_stats": self.get_quota_stats()
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
                    "quota_stats": self.get_quota_stats()
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
            text, usage, provider_used, fallback_reason = await asyncio.wait_for(
                self._generate_with_fallback(prompt),
                timeout=self.request_timeout_s
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            result = self._parse_json_text(text)
            result["ai_powered"] = True
            result["ai_meta"] = {
                "provider": provider_used,
                "model": self.providers[provider_used]["model"],
                "latency_ms": elapsed_ms,
                "token_usage": usage,
                "quota_stats": self.get_quota_stats()
            }
            if fallback_reason:
                result["ai_meta"]["fallback_reason"] = fallback_reason
            return result

        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ AI query understanding error: All providers timed out")
            return {
                "original_query": query,
                "product_type": query,
                "ai_powered": False,
                "error": "AI timeout",
                "ai_meta": {
                    "latency_ms": elapsed_ms,
                    "timeout": True,
                    "quota_stats": self.get_quota_stats()
                }
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            print(f"❌ AI query understanding error: {e}")
            return {
                "original_query": query,
                "product_type": query,
                "ai_powered": False,
                "error": str(e),
                "ai_meta": {
                    "latency_ms": elapsed_ms,
                    "timeout": False,
                    "quota_stats": self.get_quota_stats()
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
            {
                "id": i,
                "name": p.get("name", ""),
                "price": p.get("price", 0),
                "platform": p.get("platform", "")
            }
            for i, p in enumerate(products)
        ], separators=(",", ":"))

        return f'''Score each product's relevance to "{query}" from 0-100.

Products: {products_json}

Scoring guide for "{query}":
- 90-100: The product IS {query} itself (e.g., "Fresh Banana", "Toned Milk", "Basmati Rice")
- 70-89: Related {query} variant (e.g., "Raw Banana", "Full Cream Milk")  
- 40-69: {query} derivative that user might want (e.g., "Rice Flour")
- 0-39: NOT {query}, just contains the word (e.g., "Banana Chips", "Milkshake", "Dairy Milk chocolate")

Examples for "banana":
- "Fresh Yellow Banana 6pc" = 95 (IS banana)
- "Organic Banana 1kg" = 95 (IS banana)
- "Banana Chips" = 20 (processed snack)
- "Banana Shake" = 15 (beverage)

Return JSON: {{"relevant_items":[{{"id":0,"relevance_score":95,"relevance_reason":"actual banana"}}],"filtered_ids":[3,4]}}

Include in relevant_items: ALL products with score >= 50
Include in filtered_ids: ALL products with score < 50'''
    
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
                        enriched_product["relevance_score"] = self._normalize_relevance_score(
                            ai_prod.get("relevance_score", 50)
                        )
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
