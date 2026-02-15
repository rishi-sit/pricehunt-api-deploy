"""
Smart Suggestions Engine
========================
Generates dynamic, context-aware search suggestions based on:
- Query analysis and category detection
- Search history and popular searches
- Related products in the same category
- Derivative products (juice, jam, etc.)
- User's location/pincode patterns
"""

from typing import List, Dict, Any, Optional
from collections import Counter
import sqlite3
from datetime import datetime, timedelta
import json


class SuggestionsEngine:
    """
    Intelligent suggestions engine with multiple strategies:
    1. Category-based suggestions (fresh, organic, branded variants)
    2. Search history based suggestions (popular related searches)
    3. Derivative suggestions (juice, jam, chips for base items)   
    4. Smart alternatives (common confusions like grapefruit vs grape)
    """
    
    # Category-specific suggestion templates
    CATEGORY_SUGGESTIONS = {
        "fruit": {
            "keywords": ["fresh", "organic", "pack", "banana", "apple", "mango", "orange", "grapes", "strawberry", "watermelon", "papaya", "guava"],
            "variants": ["fresh", "organic", "premium", "local"],
            "derivatives": ["juice", "jam", "concentrate", "dried"],
            "brands": ["driscoll", "fresho", "freshmist", "tropicana"]
        },
        "vegetable": {
            "keywords": ["fresh", "organic", "tomato", "onion", "potato", "carrot", "spinach", "broccoli"],
            "variants": ["fresh", "organic", "local", "farm"],
            "derivatives": ["juice", "powder", "frozen"],
            "brands": ["fresho", "freshmist", "local"]
        },
        "dairy": {
            "keywords": ["milk", "curd", "butter", "yogurt", "cheese"],
            "variants": ["full cream", "toned", "organic", "lactose free"],
            "derivatives": ["yogurt", "cheese", "butter", "lassi"],
            "brands": ["amul", "mother dairy", "nandini", "verka"]
        },
        "rice": {
            "keywords": ["basmati", "sona masoori", "jasmine", "rice"],
            "variants": ["1kg", "2kg", "5kg", "organic", "basmati", "parboiled"],
            "derivatives": ["rice flour", "rice bran", "rice oil"],
            "brands": ["india gate", "daawat", "kohinoor", "aashirvaad"]
        },
        "oil": {
            "keywords": ["cooking", "sunflower", "coconut", "oil", "mustard", "olive"],
            "variants": ["1L", "5L", "10L", "virgin", "extra virgin"],
            "derivatives": ["ghee", "butter"],
            "brands": ["fortune", "saffola", "sundrop", "dhara"]
        },
        "bread": {
            "keywords": ["white", "brown", "multigrain", "bread"],
            "variants": ["whole wheat", "brown bread", "white bread", "multi grain"],
            "derivatives": ["toast", "bun", "rusk"],
            "brands": ["britannia", "harvest gold", "sunfeast"]
        },
        "eggs": {
            "keywords": ["farm fresh", "brown", "organic", "eggs", "egg"],
            "variants": ["6pcs", "12pcs", "organic", "free range", "brown", "white"],
            "derivatives": [],
            "brands": ["country eggs", "farm fresh", "nest fresh"]
        },
        "condiments": {
            "keywords": ["salt", "sugar", "sauce", "spice", "masala"],
            "variants": ["rock", "fine", "organic", "powder"],
            "derivatives": ["powder", "crystals"],
            "brands": ["tata", "catch", "everest", "badshah"]
        }
    }
    
    # Smart disambiguation for confusing searches
    DISAMBIGUATION_MAP = {
        "grape": {"exclude": ["grapefruit", "grapeseed"], "suggest": ["green grapes", "red grapes", "seedless grapes"]},
        "pine": {"exclude": ["pineapple"], "suggest": ["pine nuts"]},
        "straw": {"exclude": ["straw", "hay"], "suggest": ["strawberry", "fresh strawberry"]},
        "blue": {"exclude": ["bluebird"], "suggest": ["blueberry"]},
        "apple": {"exclude": ["apple phone", "apple mac", "apple watch", "apple iphone"], "suggest": ["fresh apple", "red apple", "green apple"]},
        "orange": {"exclude": ["orange juice"], "suggest": ["fresh orange", "sweet orange"]},
    }

    def __init__(self, db_path: Optional[str] = None):
        """Initialize suggestions engine with optional database connection."""
        self.db_path = db_path
        self.category_keywords = self._build_category_keywords()
    
    def _build_category_keywords(self) -> Dict[str, set]:
        """Build a mapping of keywords to categories."""
        keywords = {}
        for category, data in self.CATEGORY_SUGGESTIONS.items():
            for keyword in data.get("keywords", []):
                keywords[keyword.lower()] = category
        return keywords
    
    def get_category(self, query: str) -> Optional[str]:
        """Detect category from query."""
        query_lower = query.lower().strip()
        
        # Direct keyword matching
        if query_lower in self.category_keywords:
            return self.category_keywords[query_lower]
        
        # Partial matching
        for keyword, category in self.category_keywords.items():
            if keyword in query_lower:
                return category
        
        return None
    
    def generate_suggestions(
        self, 
        query: str, 
        max_suggestions: int = 5,
        include_history: bool = True,
        pincode: Optional[str] = None
    ) -> List[str]:
        """
        Generate smart suggestions for a search query.
        
        Strategy:
        1. Detect category and generate category-based suggestions
        2. Check disambiguation rules
        3. Add search history suggestions if available
        4. Mix with derivative products
        
        Args:
            query: User search query
            max_suggestions: Maximum number of suggestions to return
            include_history: Whether to include from search history
            pincode: User's pincode for location-specific suggestions
            
        Returns:
            List of suggested search queries
        """
        suggestions = []
        query_lower = query.lower().strip()
        
        # 1. Detect category and add category-based suggestions
        category = self.get_category(query)
        if category and category in self.CATEGORY_SUGGESTIONS:
            cat_data = self.CATEGORY_SUGGESTIONS[category]
            
            # Add variant suggestions (fresh, organic, etc.)
            for variant in cat_data.get("variants", [])[:2]:  # Top 2 variants
                suggestions.append(f"{query} {variant}".strip())
            
            # Add some derivative products
            for derivative in cat_data.get("derivatives", [])[:1]:  # Top 1 derivative
                suggestions.append(f"{query} {derivative}".strip())
            
            # Add brand variations for specific categories
            if category in ["dairy", "rice", "oil"]:
                for brand in cat_data.get("brands", [])[:1]:  # Top 1 brand
                    suggestions.append(f"{brand} {query}".strip())
        
        # 2. Handle disambiguation
        for base_term, disamb_rules in self.DISAMBIGUATION_MAP.items():
            if base_term in query_lower:
                # Add unambiguous suggestions
                for suggest in disamb_rules.get("suggest", [])[:2]:
                    suggestions.append(suggest)
                break
        
        # 3. Add search history suggestions if file/DB available
        if include_history and self.db_path:
            history_suggestions = self._get_history_suggestions(query_lower, category)
            suggestions.extend(history_suggestions[:2])  # Add top 2 from history
        
        # 4. Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            s_lower = s.lower()
            if s_lower not in seen and s_lower != query_lower:
                seen.add(s_lower)
                unique_suggestions.append(s)
        
        # Return top N suggestions
        return unique_suggestions[:max_suggestions]
    
    def _get_history_suggestions(self, query: str, category: Optional[str]) -> List[str]:
        """Fetch popular search suggestions from history in database."""
        if not self.db_path:
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query popular searches in the same category
            sql = """
            SELECT search_query, COUNT(*) as count
            FROM search_sessions
            WHERE query_category = ? OR search_query LIKE ?
            AND started_at > datetime('now', '-30 days')
            GROUP BY search_query
            ORDER BY count DESC
            LIMIT 5
            """
            
            cursor.execute(sql, (category, f"%{query}%"))
            results = cursor.fetchall()
            conn.close()
            
            return [row[0] for row in results if row[0].lower() != query]
        except Exception:
            # If database unavailable, return empty
            return []
    
    def get_related_products(self, query: str, max_related: int = 4) -> Dict[str, List[str]]:
        """
        Get related products and variations for a query.
        Returns grouped suggestions by type (variants, alternatives, derivatives).
        """
        category = self.get_category(query)
        
        if not category or category not in self.CATEGORY_SUGGESTIONS:
            return {"variants": [], "derivatives": [], "alternatives": []}
        
        cat_data = self.CATEGORY_SUGGESTIONS[category]
        
        return {
            "variants": [f"{query} {v}" for v in cat_data.get("variants", [])[:max_related]],
            "derivatives": [f"{query} {d}" for d in cat_data.get("derivatives", [])[:max_related]],
            "alternatives": [f"{b} {query}" for b in cat_data.get("brands", [])[:max_related]]
        }


# Singleton instance
suggestions_engine = None


def get_suggestions_engine(db_path: Optional[str] = None) -> SuggestionsEngine:
    """Get or create suggestions engine."""
    global suggestions_engine
    if suggestions_engine is None:
        suggestions_engine = SuggestionsEngine(db_path)
    return suggestions_engine
