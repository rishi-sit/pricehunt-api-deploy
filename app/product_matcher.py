"""
Product Matcher Module for PriceHunt
Matches similar products across different e-commerce platforms.

This module handles:
1. Product name normalization
2. Brand and quantity extraction
3. Cross-platform product matching
4. Best deal identification per product group
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from .ai_service import get_ai_service as get_gemini_service


@dataclass
class ProductGroup:
    """A group of matched products across platforms"""
    canonical_name: str
    brand: Optional[str]
    quantity: Optional[str]
    quantity_value: Optional[float]  # Normalized quantity (e.g., 500 for 500ml)
    quantity_unit: Optional[str]  # Normalized unit (e.g., "ml")
    products: List[Dict[str, Any]]
    best_deal: Optional[Dict[str, Any]]
    price_range: str
    savings: Optional[float] = None  # Max savings possible


@dataclass
class MatchingResult:
    """Result of product matching"""
    product_groups: List[ProductGroup]
    unmatched_products: List[Dict[str, Any]]
    ai_powered: bool
    total_products: int
    total_groups: int
    total_matched: int
    ai_meta: Optional[Dict[str, Any]] = None


class ProductMatcher:
    """
    Matches products across different platforms using AI + heuristics.
    
    Strategy:
    1. Extract brand and quantity from product names
    2. Use Gemini AI for semantic matching
    3. Fall back to rule-based matching
    4. Group by (brand, quantity) tuple
    """
    
    # Common grocery brands (for extraction)
    KNOWN_BRANDS = {
        # Dairy
        "amul", "mother dairy", "nestle", "britannia", "verka", "nandini",
        "milky mist", "go", "epigamia", "danone", "yakult",
        # Oils
        "fortune", "saffola", "sundrop", "dhara", "nature fresh", "borges",
        "figaro", "oleev", "dalda", "gemini", "freedom",
        # Rice/Grains
        "india gate", "daawat", "kohinoor", "lal qilla", "aashirvaad",
        "pillsbury", "patanjali", "24 mantra",
        # Beverages
        "tropicana", "real", "paper boat", "raw pressery", "appy fizz",
        "frooti", "maaza", "slice", "coca cola", "pepsi", "sprite", "thums up",
        # Snacks
        "lays", "kurkure", "bingo", "haldirams", "bikano", "parle",
        "britannia", "sunfeast", "hide & seek", "oreo", "bourbon",
        # Personal care
        "dove", "nivea", "himalaya", "mamaearth", "wow", "biotique",
        "patanjali", "khadi", "forest essentials",
        # Staples
        "tata", "mdh", "everest", "catch", "badshah", "saffola",
        "organic india", "24 mantra", "pro nature"
    }
    
    # Quantity patterns
    QUANTITY_PATTERNS = [
        # Weight
        (r'(\d+(?:\.\d+)?)\s*(kg|kilo|kilos|kilogram|kilograms)\b', 'kg', 1000),
        (r'(\d+(?:\.\d+)?)\s*(g|gm|gms|gram|grams)\b', 'g', 1),
        (r'(\d+(?:\.\d+)?)\s*(mg|milligram|milligrams)\b', 'mg', 0.001),
        # Volume
        (r'(\d+(?:\.\d+)?)\s*(l|lt|ltr|litre|liter|litres|liters)\b', 'l', 1000),
        (r'(\d+(?:\.\d+)?)\s*(ml|millilitre|milliliter)\b', 'ml', 1),
        # Count
        (r'(\d+)\s*(pc|pcs|piece|pieces|nos|pack|units?)\b', 'pc', 1),
        (r'pack\s+of\s+(\d+)\b', 'pc', 1),
        # Combo patterns
        (r'(\d+)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(g|ml|kg|l)\b', 'combo', 1),
    ]
    
    def __init__(self):
        self.gemini = get_gemini_service()
    
    async def match_products(
        self,
        products: List[Dict[str, Any]],
        use_ai: bool = True
    ) -> MatchingResult:
        """
        Match products across platforms.
        
        Args:
            products: List of products from multiple platforms
            use_ai: Whether to use Gemini AI for matching
            
        Returns:
            MatchingResult with grouped products
        """
        if not products:
            return MatchingResult(
                product_groups=[],
                unmatched_products=[],
                ai_powered=False,
                ai_meta=None,
                total_products=0,
                total_groups=0,
                total_matched=0
            )
        
        # Try AI-powered matching first
        ai_result = None
        print(f"🔍 ProductMatcher: use_ai={use_ai}, gemini_available={self.gemini.is_available()}, products={len(products)}")
        if use_ai and self.gemini.is_available() and len(products) > 3:
            try:
                ai_result = await self.gemini.match_products_across_platforms(products)
                print(f"🔍 ProductMatcher: gemini returned ai_powered={ai_result.get('ai_powered')}, ai_meta={ai_result.get('ai_meta')}")
            except Exception as e:
                print(f"🔍 ProductMatcher: gemini call failed with exception: {e}")
                ai_result = {
                    "ai_powered": False,
                    "ai_meta": {
                        "provider": "gemini",
                        "model": self.gemini.model_name,
                        "error": str(e)
                    },
                    "product_groups": [],
                    "unmatched_products": products
                }
        else:
            # Set default ai_meta when Gemini is skipped
            reason = "gemini_unavailable" if not self.gemini.is_available() else ("too_few_products" if len(products) <= 3 else "ai_disabled")
            ai_result = {
                "ai_powered": False,
                "ai_meta": {
                    "provider": "rule_based",
                    "model": "rule_based",
                    "reason": reason
                }
            }
            print(f"🔍 ProductMatcher: using rule-based matching, reason={reason}")

        ai_result = self._normalize_ai_result(ai_result)
        
        # Use AI results or fall back to rule-based
        ai_meta = ai_result.get("ai_meta")
        print(f"🔍 ProductMatcher: final ai_meta={ai_meta}")
        if ai_result.get("ai_powered"):
            groups = self._process_ai_groups(ai_result.get("product_groups", []))
            unmatched = ai_result.get("unmatched_products", [])
        else:
            groups, unmatched = self._rule_based_matching(products)
        
        # Convert to ProductGroup objects
        product_groups = [
            self._create_product_group(group)
            for group in groups
            if len(group.get("products", [])) >= 2  # Only groups with 2+ products
        ]
        
        # Sort groups by savings potential
        product_groups.sort(
            key=lambda g: g.savings or 0,
            reverse=True
        )
        
        total_matched = sum(len(g.products) for g in product_groups)
        
        return MatchingResult(
            product_groups=product_groups,
            unmatched_products=unmatched,
            ai_powered=ai_result.get("ai_powered", False),
            ai_meta=ai_meta,
            total_products=len(products),
            total_groups=len(product_groups),
            total_matched=total_matched
        )

    def _normalize_ai_result(self, ai_result: Any) -> Dict[str, Any]:
        if isinstance(ai_result, dict):
            return ai_result
        if isinstance(ai_result, MatchingResult):
            return {
                "ai_powered": ai_result.ai_powered,
                "ai_meta": ai_result.ai_meta,
                "product_groups": [asdict(group) for group in ai_result.product_groups],
                "unmatched_products": ai_result.unmatched_products
            }
        return {}
    
    def _rule_based_matching(
        self,
        products: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Rule-based product matching using brand and quantity extraction.
        """
        # Extract features from each product
        analyzed = []
        for product in products:
            features = self._extract_features(product)
            analyzed.append({
                "product": product,
                "features": features
            })
        
        # Group by (brand, quantity_normalized)
        groups: Dict[str, List[Dict]] = {}
        unmatched = []
        
        for item in analyzed:
            product = item["product"]
            features = item["features"]
            
            # Create grouping key
            brand = str(features.get("brand") or "").lower()
            qty_value = features.get("quantity_value")
            qty_unit = features.get("quantity_unit")
            
            if brand and qty_value:
                key = f"{brand}|{qty_value}|{qty_unit}"
            elif brand:
                # Try to match by brand + name similarity
                key = f"{brand}|{self._normalize_name(product.get('name', ''))}"
            else:
                key = None
            
            if key:
                if key not in groups:
                    groups[key] = {
                        "canonical_name": self._create_canonical_name(features, product),
                        "brand": features.get("brand"),
                        "quantity": features.get("quantity"),
                        "quantity_value": qty_value,
                        "quantity_unit": qty_unit,
                        "products": []
                    }
                
                # Only add if from different platform
                existing_platforms = {p.get("platform") for p in groups[key]["products"]}
                if product.get("platform") not in existing_platforms:
                    groups[key]["products"].append(product)
                else:
                    # Same platform, keep the cheaper one
                    for i, existing in enumerate(groups[key]["products"]):
                        if existing.get("platform") == product.get("platform"):
                            if product.get("price", float("inf")) < existing.get("price", float("inf")):
                                groups[key]["products"][i] = product
                            break
            else:
                unmatched.append(product)
        
        return list(groups.values()), unmatched
    
    def _extract_features(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract brand, quantity, and other features from product name"""
        name = str(product.get("name") or "")
        name_lower = name.lower()
        
        features = {
            "original_name": name,
            "brand": None,
            "quantity": None,
            "quantity_value": None,
            "quantity_unit": None
        }
        
        # Extract brand
        for brand in self.KNOWN_BRANDS:
            if brand in name_lower:
                features["brand"] = brand.title()
                break
        
        # If no known brand, try to extract first word as brand
        if not features["brand"]:
            words = name.split()
            if words:
                potential_brand = words[0]
                # Filter out common non-brand words
                non_brands = {"fresh", "organic", "natural", "premium", "pure", "best", "top"}
                if potential_brand.lower() not in non_brands and len(potential_brand) > 2:
                    features["brand"] = potential_brand
        
        # Extract quantity
        for pattern, unit, multiplier in self.QUANTITY_PATTERNS:
            match = re.search(pattern, name_lower, re.IGNORECASE)
            if match:
                if unit == "combo":
                    # Handle combo patterns like "2x500g"
                    count = float(match.group(1))
                    size = float(match.group(2))
                    combo_unit = match.group(3)
                    total = count * size
                    features["quantity"] = f"{int(count)}x{int(size)}{combo_unit}"
                    features["quantity_value"] = total
                    features["quantity_unit"] = combo_unit
                else:
                    value = float(match.group(1))
                    features["quantity"] = f"{match.group(1)}{unit}"
                    features["quantity_value"] = value * multiplier
                    features["quantity_unit"] = unit if unit in ["g", "ml", "pc"] else (
                        "g" if unit in ["kg", "mg"] else
                        "ml" if unit in ["l"] else
                        "pc"
                    )
                break
        
        return features
    
    def _normalize_name(self, name: str) -> str:
        """Normalize product name for comparison"""
        name = name or ""
        # Remove special characters, extra spaces
        normalized = re.sub(r'[^\w\s]', ' ', name.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove common filler words
        fillers = {"the", "a", "an", "with", "and", "for", "of", "in"}
        words = [w for w in normalized.split() if w not in fillers]
        
        return " ".join(words[:5])  # First 5 meaningful words
    
    def _create_canonical_name(
        self,
        features: Dict[str, Any],
        product: Dict[str, Any]
    ) -> str:
        """Create a canonical/standardized name for the product group"""
        parts = []
        
        if features.get("brand"):
            parts.append(features["brand"])
        
        # Extract product type from name
        name = product.get("name", "")
        # Remove brand and quantity to get product type
        product_type = name
        if features.get("brand"):
            product_type = re.sub(
                re.escape(features["brand"]),
                "",
                product_type,
                flags=re.IGNORECASE
            )
        if features.get("quantity"):
            product_type = re.sub(
                re.escape(features["quantity"]),
                "",
                product_type,
                flags=re.IGNORECASE
            )
        
        # Clean up product type
        product_type = re.sub(r'[^\w\s]', ' ', product_type)
        product_type = re.sub(r'\s+', ' ', product_type).strip()
        
        if product_type:
            # Take first few meaningful words
            words = product_type.split()[:4]
            parts.extend(words)
        
        if features.get("quantity"):
            parts.append(features["quantity"])
        
        return " ".join(parts).title()
    
    def _process_ai_groups(self, ai_groups: List[Dict]) -> List[Dict]:
        """Process AI-generated groups to ensure consistency"""
        processed = []
        
        for group in ai_groups:
            # Extract features from canonical name
            canonical = group.get("canonical_name", "")
            features = self._extract_features({"name": canonical})
            
            processed.append({
                "canonical_name": canonical,
                "brand": group.get("brand") or features.get("brand"),
                "quantity": group.get("quantity") or features.get("quantity"),
                "quantity_value": features.get("quantity_value"),
                "quantity_unit": features.get("quantity_unit"),
                "products": group.get("products", [])
            })
        
        return processed
    
    def _create_product_group(self, group: Dict) -> ProductGroup:
        """
        Create a ProductGroup object from group dict.
        
        Best Deal Selection Strategy:
        1. First, filter to only HIGH relevance products (score >= 80)
        2. Among high relevance, pick the LOWEST price
        3. If no high relevance, fall back to best relevance + lowest price
        
        This ensures "strawberry" search shows best deal on actual strawberries,
        not on strawberry-flavored items that happen to be cheaper.
        """
        products = group.get("products", [])
        
        # Find best deal (relevance-aware)
        available_products = [
            p for p in products
            if p.get("available", True) and p.get("price", 0) > 0
        ]
        
        if available_products:
            # Separate products by relevance quality
            high_relevance = [p for p in available_products if p.get("relevance_score", 50) >= 80]
            medium_relevance = [p for p in available_products if 60 <= p.get("relevance_score", 50) < 80]
            
            # Best deal selection priority:
            # 1. Lowest price among high relevance products (exact matches)
            # 2. Lowest price among medium relevance products
            # 3. Lowest price among all available products
            if high_relevance:
                # Best = highest relevance, then lowest price
                best = min(high_relevance, key=lambda x: (
                    -x.get("relevance_score", 50),  # Higher relevance first
                    x.get("price", float("inf"))    # Then lower price
                ))
                best_deal_reason = "exact_match"
            elif medium_relevance:
                best = min(medium_relevance, key=lambda x: (
                    -x.get("relevance_score", 50),
                    x.get("price", float("inf"))
                ))
                best_deal_reason = "close_match"
            else:
                best = min(available_products, key=lambda x: x.get("price", float("inf")))
                best_deal_reason = "lowest_price"
            
            prices = [p.get("price", 0) for p in available_products if p.get("price", 0) > 0]
            
            best_deal = {
                "name": best.get("name"),
                "price": best.get("price"),
                "platform": best.get("platform"),
                "url": best.get("url"),
                "image_url": best.get("image_url"),
                "relevance_score": best.get("relevance_score", 50),
                "best_deal_reason": best_deal_reason
            }
            
            min_price = min(prices)
            max_price = max(prices)
            price_range = f"₹{min_price} - ₹{max_price}" if min_price != max_price else f"₹{min_price}"
            savings = max_price - min_price if len(prices) > 1 else None
        else:
            best_deal = None
            price_range = "N/A"
            savings = None
        
        return ProductGroup(
            canonical_name=group.get("canonical_name", "Unknown Product"),
            brand=group.get("brand"),
            quantity=group.get("quantity"),
            quantity_value=group.get("quantity_value"),
            quantity_unit=group.get("quantity_unit"),
            products=products,
            best_deal=best_deal,
            price_range=price_range,
            savings=savings
        )
    
    def find_similar_products(
        self,
        target_product: Dict[str, Any],
        products: List[Dict[str, Any]],
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find products similar to a target product.
        Useful for "See similar items" feature.
        """
        target_features = self._extract_features(target_product)
        target_name_normalized = self._normalize_name(target_product.get("name", ""))
        
        scored = []
        for product in products:
            if product.get("name") == target_product.get("name"):
                continue
            
            features = self._extract_features(product)
            score = 0
            
            # Brand match
            if features.get("brand") and features["brand"] == target_features.get("brand"):
                score += 50
            
            # Quantity match
            if features.get("quantity_value") and target_features.get("quantity_value"):
                qty_ratio = features["quantity_value"] / target_features["quantity_value"]
                if 0.8 <= qty_ratio <= 1.2:  # Within 20%
                    score += 30
                elif 0.5 <= qty_ratio <= 2.0:  # Within 2x
                    score += 15
            
            # Name similarity
            name_normalized = self._normalize_name(product.get("name", ""))
            name_words = set(name_normalized.split())
            target_words = set(target_name_normalized.split())
            
            if name_words and target_words:
                overlap = len(name_words & target_words) / len(name_words | target_words)
                score += int(overlap * 20)
            
            if score > 0:
                scored.append((product, score))
        
        # Sort by score and return top results
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, s in scored[:max_results]]


# Module-level instance
_product_matcher: Optional[ProductMatcher] = None


def get_product_matcher() -> ProductMatcher:
    """Get or create ProductMatcher singleton"""
    global _product_matcher
    if _product_matcher is None:
        _product_matcher = ProductMatcher()
    return _product_matcher
