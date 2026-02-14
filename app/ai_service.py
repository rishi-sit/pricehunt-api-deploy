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
    
    def force_reset(self) -> Dict[str, Any]:
        """Force reset all quota tracking (for testing/debugging)"""
        old_stats = self.get_stats()
        self._exhausted_providers.clear()
        self._request_counts.clear()
        self._last_reset_date = self._get_today()
        print(f"🔄 FORCED quota reset - all providers now available")
        return {
            "message": "Quota reset successful",
            "previous_state": old_stats,
            "new_state": self.get_stats()
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
    PROVIDER_CEREBRAS = "cerebras"  # NEW: Fast inference, free preview
    PROVIDER_TOGETHER = "together"  # NEW: $25 free credits
    PROVIDER_OPENROUTER = "openrouter"  # NEW: Access to 100+ models
    
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
        self._setup_cerebras()  # NEW
        self._setup_together()  # NEW
        self._setup_openrouter()  # NEW: Access to 100+ models
        self._setup_together()  # NEW
        
        # Reference to global quota tracker
        self.quota = _quota_tracker
        # Track Gemini model-level exhaustion separately (per model)
        self._gemini_exhausted_models: Dict[str, str] = {}
        self._gemini_unavailable_models: Dict[str, str] = {}
        self._gemini_request_counts: Dict[str, int] = {}
        self._gemini_last_reset_date: str = self._get_today()
        
        # Check availability
        self._available = len(self.providers) > 0
        
        if self._available:
            print(f"✅ AI Service initialized")
            for name, config in self.providers.items():
                if name == self.PROVIDER_GEMINI and config.get("models"):
                    models = config.get("models") or [config.get("model")]
                    extra = f" (+{len(models) - 1} fallbacks)" if len(models) > 1 else ""
                    print(f"   ✓ {name}: {models[0]}{extra}")
                else:
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
        # HARDCODED: gemini-2.5-flash first (most reliable), gemma-3-27b-it last (often fails)
        models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemma-3-27b-it"]

        self.providers[self.PROVIDER_GEMINI] = {
            "api_key": api_key,
            "model": models[0] if models else "gemini-2.5-flash",
            "models": models,
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "type": "gemini"
        }
    
    def _setup_cerebras(self):
        """
        Setup Cerebras Cloud provider - VERY FAST inference (free preview).
        
        Cerebras offers:
        - Llama 3.3 70B at 2000+ tokens/sec (fastest inference available)
        - Free preview tier (no token limits during preview)
        - Rate limit: 30 requests/minute
        - OpenAI-compatible API
        
        Get API key at: https://cloud.cerebras.ai/
        """
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            return
        
        self.providers[self.PROVIDER_CEREBRAS] = {
            "api_key": api_key,
            "model": os.getenv("CEREBRAS_MODEL", "llama-3.3-70b"),
            "base_url": "https://api.cerebras.ai/v1",
            "type": "openai_compatible",
            "supports_json_mode": True
        }
    
    def _setup_together(self):
        """
        Setup Together.ai provider - High quality, good free tier.
        
        Together.ai offers:
        - $25 free credits for new users
        - Llama 3.1, Mixtral, Qwen, DeepSeek models
        - 60 requests/minute rate limit
        - OpenAI-compatible API
        
        Best models for product filtering:
        - meta-llama/Llama-3.3-70B-Instruct-Turbo (fast, accurate)
        - meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
        - Qwen/Qwen2.5-72B-Instruct-Turbo
        
        Get API key at: https://api.together.xyz/
        """
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            return
        
        self.providers[self.PROVIDER_TOGETHER] = {
            "api_key": api_key,
            "model": os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
            "base_url": "https://api.together.xyz/v1",
            "type": "openai_compatible",
            "supports_json_mode": True
        }
    
    def _setup_openrouter(self):
        """
        Setup OpenRouter provider - Access to 100+ AI models.
        
        OpenRouter offers:
        - Access to Claude, GPT-4, Llama, Mistral, and 100+ other models
        - Pay-per-use pricing, some free models available
        - Free credits for new users
        - Single API for multiple providers
        - OpenAI-compatible API
        
        Best free/cheap models for product filtering:
        - meta-llama/llama-3.2-3b-instruct:free (free, decent accuracy)
        - microsoft/phi-3-mini-128k-instruct:free (free, fast)
        - google/gemini-2.0-flash-exp:free (free, experimental)
        - anthropic/claude-3-haiku (cheap, very accurate)
        
        Get API key at: https://openrouter.ai/keys
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return
        
        self.providers[self.PROVIDER_OPENROUTER] = {
            "api_key": api_key,
            "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free"),
            "base_url": "https://openrouter.ai/api/v1",
            "type": "openai_compatible",
            "supports_json_mode": True,
            "extra_headers": {
                "HTTP-Referer": "https://pricehunt.app",
                "X-Title": "PriceHunt"
            }
        }
    
    def _get_available_providers(self) -> List[str]:
        """
        Get list of providers that are configured AND not quota-exhausted.
        
        Default priority (optimized for free tier + accuracy):
        1. Gemini - Good accuracy, generous free tier (1M tokens/day)
        2. Cerebras - VERY fast, unlimited during preview
        3. Together.ai - High accuracy, $25 free credits
        4. Groq - Fast, 6000 req/day free
        5. Mistral - 1B tokens/month free
        """
        default_priority = [
            self.PROVIDER_GEMINI,
            self.PROVIDER_CEREBRAS,
            self.PROVIDER_TOGETHER,
            self.PROVIDER_GROQ,
            self.PROVIDER_MISTRAL,
            self.PROVIDER_OPENROUTER  # Fallback with many free models
        ]
        env_priority = os.getenv("AI_PROVIDER_PRIORITY", "").strip()
        if env_priority:
            requested = [p.strip().lower() for p in env_priority.split(",") if p.strip()]
            priority: List[str] = []
            for p in requested:
                if p in default_priority and p not in priority:
                    priority.append(p)
            for p in default_priority:
                if p not in priority:
                    priority.append(p)
        else:
            priority = default_priority

        available = []
        for p in priority:
            if p in self.providers and self.quota.is_available(p):
                available.append(p)
        return available

    def _get_today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _check_gemini_daily_reset(self):
        today = self._get_today()
        if today != self._gemini_last_reset_date:
            self._gemini_exhausted_models.clear()
            self._gemini_unavailable_models.clear()
            self._gemini_request_counts.clear()
            self._gemini_last_reset_date = today

    def _is_gemini_model_available(self, model: str) -> bool:
        self._check_gemini_daily_reset()
        return model not in self._gemini_exhausted_models and model not in self._gemini_unavailable_models

    def _mark_gemini_model_exhausted(self, model: str):
        self._check_gemini_daily_reset()
        self._gemini_exhausted_models[model] = self._get_today()

    def _mark_gemini_model_unavailable(self, model: str):
        self._check_gemini_daily_reset()
        self._gemini_unavailable_models[model] = self._get_today()

    def _record_gemini_model_request(self, model: str):
        self._check_gemini_daily_reset()
        self._gemini_request_counts[model] = self._gemini_request_counts.get(model, 0) + 1
    
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
        if self._gemini_request_counts or self._gemini_exhausted_models or self._gemini_unavailable_models:
            stats["gemini_model_request_counts"] = dict(self._gemini_request_counts)
            stats["gemini_models_exhausted"] = list(self._gemini_exhausted_models.keys())
            stats["gemini_models_unavailable"] = list(self._gemini_unavailable_models.keys())
        return stats
    
    def force_reset_quota(self) -> Dict[str, Any]:
        """Force reset all quota tracking including Gemini model-level tracking"""
        # Reset global quota tracker
        quota_result = self.quota.force_reset()
        
        # Reset Gemini model-level tracking
        old_gemini_state = {
            "gemini_model_request_counts": dict(self._gemini_request_counts),
            "gemini_models_exhausted": list(self._gemini_exhausted_models.keys()),
            "gemini_models_unavailable": list(self._gemini_unavailable_models.keys())
        }
        self._gemini_exhausted_models.clear()
        self._gemini_unavailable_models.clear()
        self._gemini_request_counts.clear()
        self._gemini_last_reset_date = self._get_today()
        
        print(f"🔄 FORCED Gemini model quota reset - all models now available")
        
        return {
            "message": "Full quota reset successful",
            "quota_tracker_reset": quota_result,
            "gemini_previous_state": old_gemini_state,
            "new_state": self.get_quota_stats()
        }

    async def _generate_content(
        self, 
        prompt: str, 
        provider: Optional[str] = None
    ) -> tuple[str, Dict[str, Optional[int]], str, str]:
        """
        Generate content using specified provider.
        Returns: (text, usage_info, provider_used, model_used)
        """
        provider = provider or self.provider
        if not provider or provider not in self.providers:
            raise ValueError("No AI provider available")
        
        config = self.providers[provider]
        
        if config["type"] == "openai_compatible":
            text, usage = await self._generate_openai_compatible(prompt, config)
            model_used = config["model"]
        else:
            text, usage, model_used = await self._generate_gemini(prompt, config)
        
        # Record successful request
        self.quota.record_request(provider)
        
        return text, usage, provider, model_used

    async def _generate_content_override(
        self,
        prompt: str,
        provider: Optional[str],
        model_override: Optional[str]
    ) -> tuple[str, Dict[str, Optional[int]], str, str]:
        """
        Generate content using specified provider and model override.
        Returns: (text, usage_info, provider_used, model_used)
        """
        provider = provider or self.provider
        if not provider or provider not in self.providers:
            raise ValueError("No AI provider available")

        config = dict(self.providers[provider])
        model_override = (model_override or "").strip()
        if model_override:
            config["model"] = model_override
            if config.get("type") == "gemini":
                config["models"] = [model_override]

        if config["type"] == "openai_compatible":
            text, usage = await self._generate_openai_compatible(prompt, config)
            model_used = config["model"]
        else:
            text, usage, model_used = await self._generate_gemini(prompt, config)

        # Record successful request
        self.quota.record_request(provider)

        return text, usage, provider, model_used
    
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
    ) -> tuple[str, Dict[str, Optional[int]], str]:
        """Generate content using Google Gemini API"""
        models = config.get("models") or [config.get("model")]
        models_to_try = [m for m in models if m and self._is_gemini_model_available(m)]
        if not models_to_try:
            raise Exception("gemini_models_exhausted")

        last_error = None
        for model_name in models_to_try:
            model_name = model_name.strip()
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
            
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, params=params, json=payload)
                    response.raise_for_status()
                    data = response.json()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                error_text = ""
                try:
                    error_payload = e.response.json()
                    error_text = str(error_payload.get("error", {}).get("message") or "")
                except Exception:
                    error_text = (e.response.text or "").strip()
                error_text = re.sub(r"\s+", " ", error_text)[:200] if error_text else ""
                last_error = f"HTTP {status}: {error_text}" if error_text else f"HTTP {status}"

                if status == 429:
                    self._mark_gemini_model_exhausted(model_name)
                    continue
                if status in {403, 404} or (status == 400 and "model" in error_text.lower()):
                    self._mark_gemini_model_unavailable(model_name)
                    continue
                continue
            except asyncio.TimeoutError:
                last_error = "timeout"
                continue
            except Exception as e:
                last_error = str(e)
                continue
            
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
            self._record_gemini_model_request(model_name)
            return text, usage_info, model_name

        raise Exception(f"gemini_models_failed: {last_error or 'unknown_error'}")
    
    async def _generate_with_fallback(
        self, 
        prompt: str
    ) -> tuple[str, Dict[str, Optional[int]], str, str, Optional[str]]:
        """
        Generate content with automatic fallback on quota exhaustion.
        
        When a provider returns 429 (quota exceeded):
        1. Marks that provider as exhausted for today
        2. Tries the next available provider
        3. Tomorrow, will try the exhausted provider again
        
        Returns: (text, usage_info, provider_used, model_used, fallback_reason)
        """
        providers_to_try = self._get_available_providers()
        
        if not providers_to_try:
            raise Exception("All AI providers exhausted for today")
        
        last_error = None
        fallback_reason = None
        original_provider = providers_to_try[0]
        
        for i, provider in enumerate(providers_to_try):
            try:
                text, usage, used_provider, model_used = await self._generate_content(prompt, provider)
                if i > 0:
                    fallback_reason = f"Switched from {original_provider}: {last_error}"
                return text, usage, used_provider, model_used, fallback_reason
                
            except httpx.HTTPStatusError as e:
                error_text = ""
                try:
                    error_text = (e.response.text or "").strip()
                except Exception:
                    error_text = ""
                if error_text:
                    error_text = re.sub(r"\s+", " ", error_text)[:200]
                    last_error = f"HTTP {e.response.status_code}: {error_text}"
                else:
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
                if provider == self.PROVIDER_GEMINI and "gemini_models" in last_error:
                    self.quota.mark_exhausted(provider)
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
            text, usage, provider_used, model_used, fallback_reason = await asyncio.wait_for(
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
                "model": model_used,
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

    async def ping_provider(self, provider: str) -> Dict[str, Any]:
        """Connectivity test for a specific provider (no fallback)."""
        provider = (provider or "").strip().lower()
        if provider not in self.providers:
            return {
                "ok": False,
                "error": "provider_not_configured",
                "provider": provider,
                "providers_configured": list(self.providers.keys()),
                "quota_stats": self.get_quota_stats()
            }

        prompt = 'Return JSON: {"ok": true}'
        start_time = time.monotonic()
        try:
            text, usage, provider_used, model_used = await asyncio.wait_for(
                self._generate_content(prompt, provider),
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
                "provider": provider_used,
                "model": model_used,
                "token_usage": usage,
                "parsed_ok": parsed,
                "response_preview": text[:200],
                "providers_configured": list(self.providers.keys()),
                "quota_stats": self.get_quota_stats()
            }
        except httpx.HTTPStatusError as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            error_text = ""
            try:
                error_text = (e.response.text or "").strip()
            except Exception:
                error_text = ""
            if error_text:
                error_text = re.sub(r"\s+", " ", error_text)[:500]
            return {
                "ok": False,
                "latency_ms": elapsed_ms,
                "provider": provider,
                "model": self.providers[provider]["model"],
                "status_code": e.response.status_code,
                "error": error_text or "http_error",
                "quota_stats": self.get_quota_stats()
            }
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "ok": False,
                "timeout": True,
                "latency_ms": elapsed_ms,
                "provider": provider,
                "model": self.providers[provider]["model"],
                "quota_stats": self.get_quota_stats()
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "ok": False,
                "timeout": False,
                "latency_ms": elapsed_ms,
                "provider": provider,
                "model": self.providers[provider]["model"],
                "error": str(e),
                "quota_stats": self.get_quota_stats()
            }

    async def list_models(self) -> Dict[str, Any]:
        """List available Gemini models for this API key."""
        if self.PROVIDER_GEMINI not in self.providers:
            return {
                "ok": False,
                "error": "gemini_not_configured",
                "quota_stats": self.get_quota_stats()
            }

        config = self.providers[self.PROVIDER_GEMINI]
        url = f"{config['base_url']}/models"
        params = {"key": config["api_key"]}
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
            error_text = e.response.text[:500] if e.response else "No error body"
            return {
                "ok": False,
                "timeout": False,
                "latency_ms": elapsed_ms,
                "status_code": e.response.status_code,
                "error": error_text
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
            text, usage, provider_used, model_used, fallback_reason = await asyncio.wait_for(
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
                "model": model_used,
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

    async def filter_relevant_products_with_provider(
        self,
        query: str,
        products: List[Dict[str, Any]],
        strict_mode: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Filter products using a specific provider (no fallback)."""
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

        provider = (provider or "").strip().lower()
        model = (model or "").strip() or None
        if provider and provider not in self.providers:
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "error": "provider_not_configured",
                "ai_meta": {
                    "provider": provider,
                    "model": model,
                    "providers_configured": list(self.providers.keys()),
                    "quota_stats": self.get_quota_stats()
                }
            }

        products_subset = products[:self.max_input_products]
        prompt = self._build_filter_prompt(query, products_subset, strict_mode)

        text = ""
        start_time = time.monotonic()
        model_label = model or self.providers.get(provider or self.provider, {}).get("model")
        try:
            if model:
                text, usage, provider_used, model_used = await asyncio.wait_for(
                    self._generate_content_override(prompt, provider or None, model),
                    timeout=self.request_timeout_s
                )
            else:
                text, usage, provider_used, model_used = await asyncio.wait_for(
                    self._generate_content(prompt, provider or None),
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
                "model": model_used,
                "latency_ms": elapsed_ms,
                "token_usage": usage,
                "quota_stats": self.get_quota_stats()
            }
            return result

        except httpx.HTTPStatusError as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            error_text = ""
            try:
                error_text = (e.response.text or "").strip()
            except Exception:
                error_text = ""
            if error_text:
                error_text = re.sub(r"\s+", " ", error_text)[:200]
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "error": error_text or f"HTTP {e.response.status_code}",
                "ai_meta": {
                    "provider": provider or self.provider,
                    "model": model_label,
                    "latency_ms": elapsed_ms,
                    "status_code": e.response.status_code,
                    "quota_stats": self.get_quota_stats()
                }
            }
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "error": "AI timeout",
                "ai_meta": {
                    "provider": provider or self.provider,
                    "model": model_label,
                    "latency_ms": elapsed_ms,
                    "timeout": True,
                    "quota_stats": self.get_quota_stats()
                }
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "error": str(e),
                "ai_meta": {
                    "provider": provider or self.provider,
                    "model": model_label,
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
            text, usage, provider_used, model_used, fallback_reason = await asyncio.wait_for(
                self._generate_with_fallback(prompt),
                timeout=self.request_timeout_s
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            result = self._parse_json_text(text)
            result["ai_powered"] = True
            result["ai_meta"] = {
                "provider": provider_used,
                "model": model_used,
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
            text, usage, provider_used, model_used, fallback_reason = await asyncio.wait_for(
                self._generate_with_fallback(prompt),
                timeout=self.request_timeout_s
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            result = self._parse_json_text(text)
            result["ai_powered"] = True
            result["ai_meta"] = {
                "provider": provider_used,
                "model": model_used,
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
    
    def _generate_dynamic_special_notes(self, query: str, products: List[Dict]) -> str:
        """
        Generate dynamic special notes based on query patterns and actual products.
        This handles ANY query, not just hardcoded ones.
        """
        query_lower = query.lower().strip()
        query_words = query_lower.split()
        notes = []
        
        # Get product names for pattern detection
        product_names = [p.get("name", "").lower() for p in products if p.get("name")]
        product_names_str = " ".join(product_names)
        
        # ============================================================
        # 1. COMPOUND WORD TRAP DETECTION
        # Detect if products contain compound words that include the query
        # ============================================================
        compound_traps = {
            "apple": ["pineapple"],
            "grape": ["grapefruit", "grapeseed"],
            "berry": ["strawberry", "blueberry", "cranberry", "raspberry", "blackberry"],
            "orange": ["blood orange"],  # blood orange IS an orange, not a trap
            "melon": ["watermelon"],  # watermelon is different from melon
            "nut": ["coconut", "doughnut", "donut", "peanut", "walnut", "chestnut"],
            "corn": ["peppercorn", "popcorn", "acorn"],
            "lime": ["sublime"],
            "date": ["update", "outdate"],
            "fig": ["figaro", "configuration"],
            "pea": ["peanut", "peach"],
            "bean": ["jelly bean"],
        }
        
        detected_traps = []
        for base_word, trap_words in compound_traps.items():
            if base_word in query_lower:
                for trap in trap_words:
                    if trap in product_names_str and trap != query_lower:
                        detected_traps.append((base_word, trap))
        
        if detected_traps:
            trap_note = f"COMPOUND WORD TRAPS DETECTED for '{query}':\n"
            for base, trap in detected_traps:
                trap_note += f"- '{trap}' contains '{base}' but is DIFFERENT - EXCLUDE (score 0)\n"
            notes.append(trap_note)
        
        # ============================================================
        # 2. BRAND NAME TRAP DETECTION  
        # Brands that contain common product words
        # ============================================================
        brand_traps = {
            "milk": ["dairy milk", "milk bikis", "milkmaid", "milkshake"],
            "fruit": ["passion fruit", "fruit loops", "fruity"],
            "honey": ["honey bunches", "honey nut"],
            "butter": ["peanut butter", "buttermilk", "butterfly"],
            "cheese": ["cheesecake", "cheese balls"],
            "cream": ["ice cream", "cream biscuit", "cream roll"],
            "chicken": ["chicken masala", "chicken tikka"],  # These are OK
            "fish": ["fish fingers", "fish fry"],
            "egg": ["eggless", "eggplant"],
            "bread": ["breadcrumbs", "bread sticks"],
            "rice": ["rice flour", "rice bran"],
            "sugar": ["sugarcane", "sugar free"],
            "salt": ["assault", "saltine"],
            "oil": ["coil", "foil", "soil"],
            "water": ["watermelon"],
        }
        
        detected_brand_traps = []
        for product_word, traps in brand_traps.items():
            if product_word == query_lower or product_word in query_words:
                for trap in traps:
                    if trap in product_names_str:
                        detected_brand_traps.append((product_word, trap))
        
        if detected_brand_traps:
            brand_note = f"BRAND/DERIVATIVE TRAPS DETECTED for '{query}':\n"
            for base, trap in detected_brand_traps:
                brand_note += f"- '{trap}' contains '{base}' but is NOT actual {base} - EXCLUDE (score 0-20)\n"
            notes.append(brand_note)
        
        # ============================================================
        # 3. PROCESSED PRODUCT DETECTION
        # Common processed forms that should be excluded for fresh queries
        # ============================================================
        processed_suffixes = ["juice", "jam", "jelly", "shake", "smoothie", "ice cream", 
                             "chips", "powder", "flour", "oil", "vinegar", "sauce",
                             "pickle", "cake", "bread", "biscuit", "cookie", "candy",
                             "syrup", "spread", "paste", "extract", "essence", "flavour",
                             "flavored", "flavoured"]
        
        # Check if any processed forms exist in products
        processed_found = []
        for suffix in processed_suffixes:
            pattern = f"{query_lower} {suffix}"
            alt_pattern = f"{query_lower}{suffix}"  # No space
            if pattern in product_names_str or alt_pattern in product_names_str:
                processed_found.append(suffix)
        
        if processed_found:
            proc_note = f"PROCESSED PRODUCTS DETECTED for '{query}':\n"
            proc_note += f"Products containing '{query}' + [{', '.join(processed_found)}] are PROCESSED - EXCLUDE (score 10-25)\n"
            proc_note += f"Only include FRESH/RAW '{query}' products (score 85-100)\n"
            notes.append(proc_note)
        
        # ============================================================
        # 4. CATEGORY-SPECIFIC RULES
        # ============================================================
        
        # Fresh produce (fruits/vegetables) - prioritize fresh over processed
        fresh_produce = ["apple", "banana", "orange", "mango", "grape", "strawberry", 
                        "tomato", "potato", "onion", "carrot", "cucumber", "lettuce",
                        "spinach", "cabbage", "broccoli", "cauliflower", "pepper",
                        "lemon", "lime", "avocado", "kiwi", "papaya", "watermelon",
                        "pineapple", "pomegranate", "guava", "pear", "plum", "peach",
                        "cherry", "blueberry", "raspberry", "blackberry", "coconut"]
        
        if query_lower in fresh_produce or any(p in query_lower for p in fresh_produce):
            notes.append(f"""
FRESH PRODUCE RULE for '{query}':
- HIGHEST PRIORITY (95-100): Fresh, raw, whole fruit/vegetable
- MEDIUM (70-85): Cut, sliced, or packaged fresh variants  
- LOW (10-30): Processed (juice, jam, chips, dried, canned) - EXCLUDE
- ZERO (0-10): Completely different products - EXCLUDE
""")
        
        # Dairy products
        dairy_products = ["milk", "curd", "yogurt", "cheese", "butter", "ghee", "paneer", "cream"]
        if query_lower in dairy_products:
            notes.append(f"""
DAIRY PRODUCT RULE for '{query}':
- HIGHEST PRIORITY (95-100): Actual {query} product (liquid milk, fresh curd, etc.)
- MEDIUM (70-85): Variants (toned, flavored but still the base product)
- LOW (10-30): Derivatives (milkshake, cheesecake) - EXCLUDE
- ZERO (0-10): Brand names containing the word (Dairy Milk chocolate) - EXCLUDE
""")
        
        # Grains and staples
        staples = ["rice", "wheat", "flour", "atta", "dal", "lentil", "oats", "quinoa"]
        if query_lower in staples:
            notes.append(f"""
STAPLE/GRAIN RULE for '{query}':
- HIGHEST PRIORITY (95-100): Actual {query} (basmati rice, whole wheat, dal)
- MEDIUM (70-85): Close variants (brown rice, multigrain)
- LOW (20-40): Processed derivatives (rice flour, wheat biscuits) - EXCLUDE if user wants raw grain
""")
        
        # Meat/Protein
        proteins = ["chicken", "mutton", "fish", "egg", "eggs", "prawn", "shrimp", "meat"]
        if query_lower in proteins:
            notes.append(f"""
PROTEIN/MEAT RULE for '{query}':
- HIGHEST PRIORITY (95-100): Fresh/raw {query} 
- HIGH (80-95): Cleaned, cut, marinated variants
- MEDIUM (60-80): Frozen variants
- LOW (20-40): Heavily processed (nuggets, sausages) - may include based on context
""")
        
        # If no specific notes generated, add generic guidance
        if not notes:
            notes.append(f"""
GENERIC FILTERING RULE for '{query}':
- HIGHEST PRIORITY (95-100): Products that ARE exactly '{query}'
- HIGH (80-95): Close variants, same category, different brand/size
- MEDIUM (50-70): Related but different (process carefully)
- LOW (0-50): Different products that just contain the word '{query}' - EXCLUDE
""")
        
        return "\n".join(notes)

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

        query_words = query.lower().strip().split()
        is_multi_word = len(query_words) > 1
        query_lower = query.lower().strip()
        
        # STRICTER THRESHOLDS for better filtering
        # Multi-word queries need ALL words to match
        if is_multi_word:
            threshold = 75  # Increased from 60 - ALL words must match
        elif query_lower in ["milk", "rice", "oil", "sugar", "salt", "flour", "wheat", "banana", "apple", "onion", "potato", "tomato", "strawberry", "mango", "orange", "grapes", "grape", "chicken", "eggs", "bread"]:
            threshold = 85  # Higher threshold for common items with many flavored/processed variants
        else:
            threshold = 75  # Increased - better filtering for single words

        # Build special notes DYNAMICALLY based on query patterns
        special_notes = self._generate_dynamic_special_notes(query_lower, products)

        multi_word_note = ""
        if is_multi_word:
            # Extract key words for better matching
            key_words = [w for w in query_words if len(w) > 2]  # Ignore short words like "a", "an", "the"
            multi_word_note = f"""
CRITICAL: The user searched for "{query}" (multi-word query with {len(key_words)} key words: {', '.join(key_words)}).
This is a SPECIFIC search — ALL key words MUST be present or closely matched in the product name.
Products that match only SOME words should score LOW (0-40).

STRICT MATCHING RULES:
- Score 90-100: ALL key words present, exact match or very close variant
- Score 70-89: ALL key words present but different order or slight variation (ONLY if still the same product)
- Score 40-69: Missing 1 key word or wrong variant - FILTER OUT
- Score 0-39: Missing 2+ key words or completely different product - FILTER OUT

Examples for "milk double toned":
- "Amul Double Toned Milk 500ml" = 95 ✓ (has "milk", "double", "toned" - ALL words match)
- "Mother Dairy Double Toned Milk 1L" = 95 ✓ (has all words, correct product)
- "Amul Toned Milk 500ml" = 35 ✗ (missing "double" - wrong variant, FILTER OUT)
- "Amul Full Cream Milk 1L" = 20 ✗ (missing "double toned" - wrong variant, FILTER OUT)
- "Cadbury Dairy Milk Chocolate" = 5 ✗ (not milk at all, FILTER OUT)

Examples for "banana yellow fresh":
- "Fresh Yellow Banana 6pc" = 95 ✓ (has "banana", "yellow", "fresh" - ALL words match)
- "Yellow Banana 1kg" = 70 ✓ (has "banana" and "yellow", "fresh" implied - acceptable)
- "Green Banana 1kg" = 25 ✗ (missing "yellow" - wrong variant, FILTER OUT)
- "Banana Chips" = 10 ✗ (processed product, not fresh banana, FILTER OUT)

REMEMBER: For multi-word queries, be STRICT. Missing even ONE key word means the product is NOT what the user wants.
"""

        return f'''You are an expert product relevance filter. Score each product's relevance to the user's search query "{query}" from 0-100.

Products to score: {products_json}
{milk_note}
{special_notes}
{multi_word_note}

SCORING GUIDELINES FOR "{query}":
- 90-100: The product IS exactly "{query}" (matches ALL query words, is the actual product the user wants)
- 75-89: Very close variant (same product type, all key words present, minor differences like brand or size)
- 50-74: Related but different variant (missing key words or wrong type/flavor) - FILTER OUT
- 25-49: Partially related but NOT what user wants - FILTER OUT
- 0-24: Completely irrelevant, just contains some words from query - FILTER OUT

CRITICAL RULES:
1. For multi-word queries: ALL key words must be present. Missing even ONE key word = score < 50 (FILTER OUT)
2. For single-word queries: Be strict about derivatives. "milk chocolate" is NOT "milk"
3. Processed/derived products are NOT the same as the base product (e.g., "banana chips" ≠ "banana")
4. Wrong variants/flavors/types should be filtered out (e.g., "full cream milk" ≠ "double toned milk")
5. PRIORITIZE exact matches: Fresh fruits/vegetables score HIGHEST (95-100), processed versions score LOW

EXAMPLES FOR SINGLE-WORD QUERIES:
Query: "strawberry"
- "Fresh Strawberry 200g" = 100 ✓ (IS strawberry fruit - HIGHEST priority)
- "Strawberry Punnet 250g" = 98 ✓ (IS strawberry fruit)
- "Organic Strawberries 500g" = 98 ✓ (IS strawberry fruit)
- "Strawberry Jam" = 25 ✗ (processed - FILTER OUT)
- "Strawberry Yogurt" = 20 ✗ (flavored product - FILTER OUT)
- "Strawberry Cake" = 15 ✗ (baked product - FILTER OUT)
- "Strawberry Ice Cream" = 15 ✗ (flavored product - FILTER OUT)
- "Strawberry Shake" = 10 ✗ (beverage - FILTER OUT)

Query: "banana"
- "Fresh Yellow Banana 6pc" = 100 ✓ (IS banana fruit - HIGHEST priority)
- "Organic Banana 1kg" = 98 ✓ (IS banana fruit)
- "Robusta Banana" = 98 ✓ (IS banana fruit)
- "Banana Chips" = 20 ✗ (processed snack - FILTER OUT)
- "Banana Shake" = 15 ✗ (beverage - FILTER OUT)
- "Banana Bread" = 10 ✗ (baked product - FILTER OUT)

Query: "apple"
- "Fresh Apple 1kg" = 100 ✓ (IS apple fruit - HIGHEST priority)
- "Shimla Apple 4pc" = 98 ✓ (IS apple fruit)
- "Washington Apple" = 98 ✓ (IS apple fruit)
- "Apple Juice" = 25 ✗ (processed beverage - FILTER OUT)
- "Apple Cider Vinegar" = 15 ✗ (vinegar - FILTER OUT)
- "Apple Pie" = 10 ✗ (baked product - FILTER OUT)

Query: "rice"
- "Basmati Rice 1kg" = 95 ✓ (IS rice)
- "Brown Rice 500g" = 95 ✓ (IS rice)
- "Rice Flour" = 25 ✗ (processed product - FILTER OUT)
- "Rice Cakes" = 15 ✗ (processed snack - FILTER OUT)

EXAMPLES FOR MULTI-WORD QUERIES:
Query: "milk double toned"
- "Amul Double Toned Milk 500ml" = 95 ✓ (has "milk", "double", "toned" - ALL words)
- "Mother Dairy Double Toned Milk 1L" = 95 ✓ (ALL words present)
- "Amul Toned Milk 500ml" = 35 ✗ (missing "double" - FILTER OUT)
- "Amul Full Cream Milk" = 20 ✗ (missing "double toned" - FILTER OUT)

Query: "banana yellow fresh"
- "Fresh Yellow Banana 6pc" = 95 ✓ (has "banana", "yellow", "fresh" - ALL words)
- "Yellow Banana 1kg" = 75 ✓ (has "banana" and "yellow", "fresh" implied)
- "Green Banana 1kg" = 30 ✗ (missing "yellow" - FILTER OUT)
- "Banana Chips" = 10 ✗ (not fresh banana - FILTER OUT)

OUTPUT FORMAT:
Return ONLY valid JSON:
{{"relevant_items":[{{"id":0,"relevance_score":95,"relevance_reason":"exact match - all words present"}}],"filtered_ids":[3,4]}}

STRICT FILTERING RULE: 
- Include in relevant_items ONLY products with score >= {threshold}
- Include in filtered_ids: ALL products with score < {threshold}
- Be STRICT - it's better to filter out a borderline product than show irrelevant results'''
    
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
