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
from typing import List, Dict, Optional, Any
import google.generativeai as genai
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
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("⚠️ GEMINI_API_KEY not set - AI features will be disabled")
            self.model = None
            return
            
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 2.5 Flash for speed and cost efficiency
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0.1,  # Low temperature for consistent results
                "top_p": 0.95,
                "max_output_tokens": 4096,
                "response_mime_type": "application/json"  # Force JSON output
            }
        )
        print("✅ Gemini AI initialized (model: gemini-2.5-flash)")
    
    def is_available(self) -> bool:
        """Check if Gemini is properly configured"""
        return self.model is not None
    
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
                "ai_powered": False
            }
        
        # Limit products to avoid token limits (prioritize by price)
        sorted_products = sorted(products, key=lambda x: x.get("price", float("inf")))
        products_subset = sorted_products[:100]  # Max 100 products
        
        # Build the prompt
        prompt = self._build_filter_prompt(query, products_subset, strict_mode)
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            result = json.loads(response.text)
            result["ai_powered"] = True
            
            # Map back to full product data
            result["relevant_products"] = self._enrich_products(
                result.get("relevant_products", []), 
                products
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Gemini filter error: {e}")
            # Fallback on error
            return {
                "relevant_products": products,
                "filtered_out": [],
                "query_understanding": {"original": query, "interpreted_as": query},
                "ai_powered": False,
                "error": str(e)
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
                "ai_powered": False
            }
        
        # Limit products
        products_subset = products[:80]
        
        prompt = self._build_matching_prompt(products_subset)
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            result = json.loads(response.text)
            result["ai_powered"] = True
            
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
            
        except Exception as e:
            print(f"❌ Gemini matching error: {e}")
            return {
                "product_groups": [],
                "unmatched_products": products,
                "ai_powered": False,
                "error": str(e)
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
                "ai_powered": False
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
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            result = json.loads(response.text)
            result["ai_powered"] = True
            return result
            
        except Exception as e:
            print(f"❌ Gemini query understanding error: {e}")
            return {
                "original_query": query,
                "product_type": query,
                "ai_powered": False,
                "error": str(e)
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
        ], indent=2)
        
        strictness = """
STRICT FILTERING RULES:
- For single word searches like "milk", ONLY include actual milk products
- "Milkmaid", "Milkshake", "Dairy Milk chocolate" are NOT milk - filter them out
- "Amul Milk", "Mother Dairy Milk", "Toned Milk" ARE milk - include them
- If the search term appears as part of a BRAND NAME but the product is different, filter it out
- Example: Search "grape" should NOT include "Grapefruit" (different fruit)
""" if strict_mode else """
LENIENT FILTERING:
- Include products where the search term is the main ingredient or flavor
- Only filter obviously unrelated products
"""
        
        return f"""You are a smart grocery search filter. Given a search query and product list, 
identify which products are ACTUALLY what the user is looking for.

SEARCH QUERY: "{query}"

{strictness}

PRODUCTS TO FILTER:
{products_json}

Return a JSON object with:
{{
  "query_understanding": {{
    "original": "{query}",
    "interpreted_as": "what the user actually wants",
    "category": "product category"
  }},
  "relevant_products": [
    {{
      "name": "product name",
      "price": price,
      "platform": "platform",
      "relevance_score": 0-100,
      "relevance_reason": "why this is relevant"
    }}
  ],
  "filtered_out": [
    {{
      "name": "product name", 
      "platform": "platform",
      "filter_reason": "why filtered out"
    }}
  ]
}}

Sort relevant_products by relevance_score (highest first), then by price (lowest first).
Be STRICT - only include products that genuinely match the search intent.

JSON only:"""
    
    def _build_matching_prompt(self, products: List[Dict]) -> str:
        """Build the prompt for product matching"""
        
        products_json = json.dumps([
            {
                "name": p.get("name", ""),
                "price": p.get("price", 0),
                "platform": p.get("platform", "")
            }
            for p in products
        ], indent=2)
        
        return f"""You are a product matching expert. Group these products from different platforms 
that are THE SAME PRODUCT (same brand, same size, same variant).

PRODUCTS:
{products_json}

MATCHING RULES:
1. Only match products that are EXACTLY the same (same brand, size, variant)
2. "Amul Toned Milk 500ml" from Zepto = "Amul Taaza 500ml" from BigBasket (same product, different naming)
3. "Amul Toned Milk 500ml" ≠ "Amul Toned Milk 1L" (different sizes)
4. "Amul Butter 100g" ≠ "Mother Dairy Butter 100g" (different brands)
5. Normalize names - platforms use different formats

Return JSON:
{{
  "product_groups": [
    {{
      "canonical_name": "standardized product name with brand and size",
      "brand": "brand name",
      "quantity": "quantity/size",
      "products": [
        {{"name": "original name", "price": price, "platform": "platform"}}
      ],
      "best_deal": {{"platform": "cheapest platform", "price": lowest_price}},
      "price_range": "₹XX - ₹YY"
    }}
  ],
  "unmatched_products": [
    {{"name": "name", "price": price, "platform": "platform"}}
  ]
}}

Only create groups with 2+ products from DIFFERENT platforms.
Products with no match go in unmatched_products.

JSON only:"""
    
    def _enrich_products(
        self, 
        ai_products: List[Dict], 
        original_products: List[Dict]
    ) -> List[Dict]:
        """Enrich AI-returned products with full original data"""
        enriched = []
        
        for ai_prod in ai_products:
            # Find matching original product
            name = ai_prod.get("name", "").lower()
            platform = ai_prod.get("platform", "").lower()
            
            for orig in original_products:
                orig_name = orig.get("name", "").lower()
                orig_platform = orig.get("platform", "").lower()
                
                # Match by name similarity and platform
                if platform in orig_platform or orig_platform in platform:
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
