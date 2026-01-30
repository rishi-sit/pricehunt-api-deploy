"""
Smart Search Module for PriceHunt
Combines Gemini AI with rule-based filtering for optimal search results.

This module handles:
1. Query preprocessing and understanding
2. AI-powered relevance filtering
3. Fallback rule-based filtering when AI is unavailable
4. Result ranking and sorting
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .gemini_service import get_gemini_service


@dataclass
class SearchResult:
    """Search result with relevance information"""
    products: List[Dict[str, Any]]
    filtered_out: List[Dict[str, Any]]
    query_understanding: Dict[str, Any]
    best_deal: Optional[Dict[str, Any]]
    ai_powered: bool
    total_found: int
    total_filtered: int


class SmartSearch:
    """
    Smart search engine that combines AI and rule-based filtering.
    
    Strategy:
    1. Try Gemini AI for semantic understanding
    2. Fall back to rule-based filtering if AI fails
    3. Apply post-processing rules for edge cases
    """
    
    # Words that indicate a derivative/processed product
    DERIVATIVE_INDICATORS = {
        "juice", "squash", "shake", "smoothie", "cocktail",
        "jam", "jelly", "preserve", "marmalade",
        "sauce", "ketchup", "puree", "paste", "chutney", "pickle",
        "syrup", "concentrate", "extract", "essence",
        "powder", "dried", "flakes", "flour",
        "chips", "crisps", "wafers", "fries",
        "candy", "toffee", "chocolate", "fudge",
        "cake", "pastry", "muffin", "cookie", "biscuit",
        "ice cream", "icecream", "kulfi", "gelato",
        "flavour", "flavor", "flavored", "flavoured",
        "soap", "shampoo", "lotion", "cream", "moisturizer"
    }
    
    # Product types that should NOT be filtered (user is searching for these)
    PRODUCT_TYPES = {
        "juice", "oil", "flour", "powder", "sauce", "jam", "pickle",
        "chips", "biscuit", "chocolate", "bread", "rice", "dal",
        "sugar", "salt", "tea", "coffee", "butter", "ghee", "cheese"
    }
    
    # Compound word traps (search term vs compound word)
    COMPOUND_TRAPS = {
        "grape": ["grapefruit", "grapeseed"],
        "pine": ["pineapple"],
        "straw": ["strawberry"],
        "blue": ["blueberry"],
        "black": ["blackberry", "blackcurrant"],
        "coco": ["coconut"],
        "butter": ["buttermilk", "butterfly"],
        "sun": ["sunflower"],
        "ground": ["groundnut"],
        "corn": ["cornflakes", "popcorn"],  # Note: popcorn is different from corn
    }
    
    def __init__(self):
        self.gemini = get_gemini_service()
    
    async def search(
        self,
        query: str,
        products: List[Dict[str, Any]],
        strict_mode: bool = True
    ) -> SearchResult:
        """
        Perform smart search with AI + rule-based filtering.
        
        Args:
            query: User's search query
            products: List of scraped products from all platforms
            strict_mode: Whether to be strict about filtering
            
        Returns:
            SearchResult with filtered products and metadata
        """
        if not products:
            return SearchResult(
                products=[],
                filtered_out=[],
                query_understanding={"original": query, "interpreted_as": query},
                best_deal=None,
                ai_powered=False,
                total_found=0,
                total_filtered=0
            )
        
        # Normalize query
        query_normalized = query.lower().strip()
        query_words = query_normalized.split()
        primary_keyword = query_words[0] if query_words else query_normalized
        
        # Check if this is a product type search (don't filter derivatives)
        is_product_type_search = primary_keyword in self.PRODUCT_TYPES
        
        # Try AI-powered filtering first
        ai_result = None
        if self.gemini.is_available() and len(products) > 5:
            ai_result = await self.gemini.filter_relevant_products(
                query, 
                products,
                strict_mode=strict_mode and not is_product_type_search
            )

        ai_result = self._normalize_ai_result(ai_result)
        
        # Apply rule-based filtering
        if ai_result.get("ai_powered"):
            relevant = ai_result.get("relevant_products", [])
            filtered = ai_result.get("filtered_out", [])
            query_understanding = ai_result.get("query_understanding", {})
        else:
            # Fallback to rule-based filtering
            relevant, filtered = self._rule_based_filter(
                query_normalized,
                primary_keyword,
                products,
                is_product_type_search
            )
            query_understanding = {
                "original": query,
                "interpreted_as": query,
                "is_product_type": is_product_type_search
            }
        
        # Post-process: apply additional rules
        relevant, extra_filtered = self._post_process(
            relevant, 
            query_normalized,
            primary_keyword,
            is_product_type_search
        )
        filtered.extend(extra_filtered)
        
        # Sort by relevance score (if available) then price
        relevant = sorted(
            relevant,
            key=lambda x: (-x.get("relevance_score", 50), x.get("price", float("inf")))
        )
        
        # Find best deal
        best_deal = self._find_best_deal(relevant, query_normalized)
        
        return SearchResult(
            products=relevant,
            filtered_out=filtered,
            query_understanding=query_understanding,
            best_deal=best_deal,
            ai_powered=ai_result.get("ai_powered", False),
            total_found=len(relevant),
            total_filtered=len(filtered)
        )

    def _normalize_ai_result(self, ai_result: Any) -> Dict[str, Any]:
        if isinstance(ai_result, dict):
            return ai_result
        if isinstance(ai_result, SearchResult):
            return {
                "ai_powered": ai_result.ai_powered,
                "relevant_products": ai_result.products,
                "filtered_out": ai_result.filtered_out,
                "query_understanding": ai_result.query_understanding
            }
        return {}
    
    def _rule_based_filter(
        self,
        query: str,
        primary_keyword: str,
        products: List[Dict],
        is_product_type_search: bool
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Rule-based filtering when AI is unavailable.
        """
        relevant = []
        filtered = []
        
        for product in products:
            name = product.get("name", "").lower()
            
            # Check if product is relevant
            is_relevant, reason = self._is_relevant(
                name, 
                query, 
                primary_keyword,
                is_product_type_search
            )
            
            if is_relevant:
                product["relevance_score"] = self._calculate_relevance_score(
                    name, query, primary_keyword
                )
                product["relevance_reason"] = reason
                relevant.append(product)
            else:
                filtered.append({
                    "name": product.get("name"),
                    "platform": product.get("platform"),
                    "filter_reason": reason
                })
        
        return relevant, filtered
    
    def _is_relevant(
        self,
        product_name: str,
        query: str,
        primary_keyword: str,
        is_product_type_search: bool
    ) -> Tuple[bool, str]:
        """
        Check if a product is relevant to the search query.
        Returns (is_relevant, reason)
        """
        name_lower = product_name.lower()
        
        # 1. Check for compound word traps
        if primary_keyword in self.COMPOUND_TRAPS:
            for trap in self.COMPOUND_TRAPS[primary_keyword]:
                if trap in name_lower:
                    return False, f"Compound word: '{trap}' is different from '{primary_keyword}'"
        
        # 2. Check if keyword exists as a complete word
        keyword_pattern = rf'\b{re.escape(primary_keyword)}s?\b'
        has_exact_match = bool(re.search(keyword_pattern, name_lower))
        
        if not has_exact_match:
            # Keyword doesn't appear as a word
            if primary_keyword in name_lower:
                # It's part of another word
                return False, f"'{primary_keyword}' is part of another word, not the product"
            return False, f"Does not contain '{primary_keyword}'"
        
        # 3. For non-product-type searches, check for derivatives
        if not is_product_type_search:
            for indicator in self.DERIVATIVE_INDICATORS:
                if indicator in name_lower:
                    # Check if the derivative IS the search term
                    if indicator == primary_keyword or indicator == query:
                        continue  # User is searching for this derivative
                    return False, f"Derivative product ({indicator})"
        
        # 4. Check if keyword is used as modifier/flavor
        is_modifier = self._is_keyword_modifier(name_lower, primary_keyword)
        if is_modifier and not is_product_type_search:
            return False, f"'{primary_keyword}' is used as flavor/modifier, not the main product"
        
        return True, "Matches search criteria"
    
    def _is_keyword_modifier(self, product_name: str, keyword: str) -> bool:
        """
        Check if the keyword is used as a modifier (flavor/type) rather than the main product.
        E.g., "mango ice cream" - mango is modifier, ice cream is product
        """
        # Product type words that would come AFTER a modifier
        product_types = [
            "juice", "shake", "smoothie", "ice cream", "icecream",
            "cake", "pie", "jam", "jelly", "sauce", "chutney",
            "candy", "chocolate", "biscuit", "cookie", "chips"
        ]
        
        # Find keyword position
        try:
            keyword_pos = product_name.index(keyword)
        except ValueError:
            return False
        
        # Check if any product type appears after the keyword
        for ptype in product_types:
            if ptype in product_name:
                ptype_pos = product_name.index(ptype)
                if ptype_pos > keyword_pos:
                    return True
        
        return False
    
    def _calculate_relevance_score(
        self,
        product_name: str,
        query: str,
        primary_keyword: str
    ) -> int:
        """Calculate relevance score (0-100) for a product"""
        name_lower = product_name.lower()
        words = name_lower.split()
        score = 50  # Base score
        
        # +30: Name starts with keyword
        if words and words[0].startswith(primary_keyword):
            score += 30
        # +20: Keyword in first 3 words
        elif any(w.startswith(primary_keyword) for w in words[:3]):
            score += 20
        # +10: Keyword appears somewhere
        elif primary_keyword in name_lower:
            score += 10
        
        # +10: Has "fresh" or "organic"
        if "fresh" in name_lower or "organic" in name_lower:
            score += 10
        
        # +5: Has quantity indicator
        if re.search(r'\d+\s*(g|gm|kg|ml|l|pc|pcs)\b', name_lower):
            score += 5
        
        # -10: Very long name (might be a combo/bundle)
        if len(product_name) > 80:
            score -= 10
        
        # -5: Contains "combo" or "pack of" (unless searched for)
        if "combo" in name_lower and "combo" not in query:
            score -= 5
        
        return min(100, max(0, score))
    
    def _post_process(
        self,
        products: List[Dict],
        query: str,
        primary_keyword: str,
        is_product_type_search: bool
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Post-process filtered results for edge cases.
        """
        final_relevant = []
        extra_filtered = []
        
        for product in products:
            name = product.get("name", "").lower()
            
            # Additional check: brand name contains keyword but product is different
            # E.g., "Dairy Milk" chocolate when searching "milk"
            if self._is_brand_name_trap(name, primary_keyword) and not is_product_type_search:
                extra_filtered.append({
                    "name": product.get("name"),
                    "platform": product.get("platform"),
                    "filter_reason": f"'{primary_keyword}' is part of brand name, not the product"
                })
                continue
            
            final_relevant.append(product)
        
        return final_relevant, extra_filtered
    
    def _is_brand_name_trap(self, product_name: str, keyword: str) -> bool:
        """
        Check if keyword appears in brand name but product is different.
        E.g., "Cadbury Dairy Milk" - "milk" is in brand name but it's chocolate
        """
        brand_traps = {
            "milk": ["dairy milk", "milkmaid", "milky bar", "milky way"],
            "fruit": ["fruit loops", "fruity"],
            "gold": ["gold flake", "gold star"],
            "sun": ["sunfeast", "sundrop"],
        }
        
        if keyword in brand_traps:
            for trap in brand_traps[keyword]:
                if trap in product_name:
                    return True
        
        return False
    
    def _find_best_deal(
        self, 
        products: List[Dict],
        query: str
    ) -> Optional[Dict[str, Any]]:
        """Find the best deal among relevant products"""
        if not products:
            return None
        
        # Prefer highly relevant products
        high_relevance = [p for p in products if p.get("relevance_score", 0) >= 70]
        candidates = high_relevance if high_relevance else products
        
        # Filter available products
        available = [p for p in candidates if p.get("available", True) and p.get("price", 0) > 0]
        
        if not available:
            return None
        
        # Find cheapest
        best = min(available, key=lambda x: x.get("price", float("inf")))
        
        return {
            "name": best.get("name"),
            "price": best.get("price"),
            "platform": best.get("platform"),
            "url": best.get("url"),
            "image_url": best.get("image_url"),
            "relevance_score": best.get("relevance_score", 50)
        }


# Module-level instance
_smart_search: Optional[SmartSearch] = None


def get_smart_search() -> SmartSearch:
    """Get or create SmartSearch singleton"""
    global _smart_search
    if _smart_search is None:
        _smart_search = SmartSearch()
    return _smart_search
