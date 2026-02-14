"""
FastAPI server to provide product search API for Android app.
Uses the existing scrapers with Playwright to bypass anti-bot protection.

NEW in v2.0:
- Gemini AI-powered smart search filtering
- Cross-platform product matching
- Natural language query understanding
"""
from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import asyncio
import time
from app.scrapers.amazon import AmazonScraper
from app.scrapers.amazon_fresh import AmazonFreshScraper
from app.scrapers.flipkart import FlipkartScraper
from app.scrapers.flipkart_minutes import FlipkartMinutesScraper
from app.scrapers.bigbasket import BigBasketScraper
from app.scrapers.jiomart import JioMartScraper
from app.scrapers.jiomart_quick import JioMartQuickScraper
from app.scrapers.zepto import ZeptoScraper
from app.scrapers.blinkit import BlinkitScraper
from app.scrapers.instamart import InstamartScraper

# AI-powered modules
from app.smart_search import get_smart_search
from app.product_matcher import get_product_matcher
from app.ai_service import get_ai_service as get_gemini_service

MAX_PLATFORM_ITEMS = 10


# Request/Response models
class ProductInput(BaseModel):
    """Product data from Android scraping"""
    name: str
    price: float
    original_price: Optional[float] = None
    discount: Optional[str] = None
    platform: str
    url: Optional[str] = None
    image_url: Optional[str] = None
    rating: Optional[float] = None
    delivery_time: Optional[str] = None
    available: bool = True


class SmartSearchRequest(BaseModel):
    """Request body for smart search"""
    query: str
    products: List[ProductInput] = []
    pincode: Optional[str] = "560001"
    strict_mode: bool = True
    platform_results: Optional[Dict[str, List[ProductInput]]] = None


class MatchProductsRequest(BaseModel):
    """Request body for product matching"""
    products: List[ProductInput]


app = FastAPI(
    title="PriceHunt API",
    version="2.0.0",
    description="AI-powered price comparison API with smart search and product matching"
)

# Allow CORS for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize scrapers
scrapers = {}

def get_scrapers(pincode: str):
    """Get or create scrapers for the given pincode."""
    if pincode not in scrapers:
        scrapers[pincode] = {
            "Amazon Fresh": AmazonFreshScraper(pincode),
            "Flipkart Minutes": FlipkartMinutesScraper(pincode),
            "JioMart Quick": JioMartQuickScraper(pincode),
            "BigBasket": BigBasketScraper(pincode),
            "Zepto": ZeptoScraper(pincode),
            "Amazon": AmazonScraper(pincode),
            "Flipkart": FlipkartScraper(pincode),
            "JioMart": JioMartScraper(pincode),
            "Blinkit": BlinkitScraper(pincode),
            "Instamart": InstamartScraper(pincode),
        }
    return scrapers[pincode]


@app.get("/api/search")
async def search_products(
    q: str = Query(..., description="Search query"),
    pincode: str = Query("560001", description="Delivery pincode")
) -> Dict:
    """
    Search for products across all platforms.
    Returns results as they become available.
    """
    platform_scrapers = get_scrapers(pincode)
    results = {}
    
    # Run all scrapers in parallel
    tasks = []
    for platform_name, scraper in platform_scrapers.items():
        tasks.append(search_platform(platform_name, scraper, q))
    
    # Wait for all to complete
    platform_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results into a flat list
    all_products = []
    for platform_name, products in zip(platform_scrapers.keys(), platform_results):
        if not isinstance(products, Exception) and products:
            for p in products:
                all_products.append({
                    "name": p.name,
                    "price": p.price,
                    "original_price": p.original_price,
                    "discount": p.discount,
                    "platform": p.platform,
                    "url": p.url,
                    "image_url": p.image_url,
                    "rating": p.rating,
                    "delivery_time": p.delivery_time,
                    "available": p.available
                })
    
    # Find lowest price
    lowest = None
    if all_products:
        lowest = min(all_products, key=lambda x: x["price"])
    
    return {
        "query": q,
        "pincode": pincode,
        "results": all_products,
        "lowest_price": lowest,
        "total_platforms": len([p for p in platform_results if not isinstance(p, Exception) and p])
    }


async def search_platform(platform_name: str, scraper, query: str):
    """Search a single platform."""
    try:
        print(f"{platform_name}: Searching for '{query}'...")
        products = await scraper.search(query)
        print(f"{platform_name}: Found {len(products)} products")
        return products
    except Exception as e:
        print(f"{platform_name}: Error - {e}")
        return []


@app.get("/api/platforms")
async def get_platforms():
    """Get list of all supported platforms."""
    return {
        "platforms": [
            {"name": "Amazon Fresh", "delivery_time": "2-4 hours"},
            {"name": "Flipkart Minutes", "delivery_time": "10-45 mins"},
            {"name": "JioMart Quick", "delivery_time": "10-30 mins"},
            {"name": "BigBasket", "delivery_time": "2-4 hours"},
            {"name": "Zepto", "delivery_time": "10-15 mins"},
            {"name": "Amazon", "delivery_time": "1-3 days"},
            {"name": "Flipkart", "delivery_time": "2-4 days"},
            {"name": "JioMart", "delivery_time": "2-5 days"},
            {"name": "Blinkit", "delivery_time": "10-20 mins"},
            {"name": "Instamart", "delivery_time": "15-30 mins"},
        ]
    }


@app.get("/")
async def root():
    """API root endpoint."""
    gemini = get_gemini_service()
    return {
        "message": "PriceHunt API - AI-Powered Price Comparison",
        "version": "2.0.0",
        "ai_enabled": gemini.is_available(),
        "endpoints": {
            "/api/search": "Search products across platforms (server-side scraping)",
            "/api/smart-search": "🆕 AI-powered smart search (for Android scraped data)",
            "/api/match-products": "🆕 Match similar products across platforms",
            "/api/understand-query": "🆕 Understand search query intent",
            "/api/platforms": "Get supported platforms",
            "/api/groq-ping": "Groq connectivity test (no fallback)",
            "/docs": "API documentation"
        }
    }


# ============================================================================
# NEW AI-POWERED ENDPOINTS
# ============================================================================

@app.post("/api/smart-search")
async def smart_search(request: SmartSearchRequest):
    """
    🆕 AI-powered smart search filtering.
    
    Takes scraped products from Android app and filters them using Gemini AI
    to return only relevant results.
    
    Example:
    - Query: "milk"
    - Input: [Amul Milk, Milkmaid, Dairy Milk Chocolate, Mother Dairy Milk]
    - Output: [Amul Milk, Mother Dairy Milk] (filtered out non-milk products)
    """
    smart_search_service = get_smart_search()
    start_time = time.monotonic()
    
    # Convert Pydantic models to dicts (prefer platform-wise input if provided)
    platform_results = request.platform_results or {}
    if platform_results:
        limited_platform_results = {
            platform: items[:MAX_PLATFORM_ITEMS]
            for platform, items in platform_results.items()
        }
        products = [
            p.model_dump()
            for platform_products in limited_platform_results.values()
            for p in platform_products
        ]
    else:
        products = [p.model_dump() for p in request.products]
    
    filter_start = time.monotonic()
    result = await smart_search_service.search(
        query=request.query,
        products=products,
        strict_mode=request.strict_mode,
        use_ai=True,
        ai_skip_reason=None
    )
    filter_ms = int((time.monotonic() - filter_start) * 1000)
    total_ms = int((time.monotonic() - start_time) * 1000)
    
    platform_counts = {
        platform: len(items) for platform, items in limited_platform_results.items()
    } if platform_results else {}

    return {
        "query": request.query,
        "pincode": request.pincode,
        "ai_powered": result.ai_powered,
        "ai_meta": result.ai_meta,
        "query_understanding": result.query_understanding,
        "results": result.products,
        "filtered_out": result.filtered_out,
        "best_deal": result.best_deal,
        "timing_ms": {
            "total": total_ms,
            "filter": filter_ms
        },
        "stats": {
            "total_input": len(products),
            "total_relevant": result.total_found,
            "total_filtered": result.total_filtered,
            "platform_counts": platform_counts
        }
    }


@app.post("/api/match-products")
async def match_products(request: MatchProductsRequest):
    """
    🆕 Match similar products across platforms.
    
    Groups products that are the same item (same brand, size) from different
    platforms to show price comparison.
    
    Example:
    - Input: [Amul Milk 500ml @Zepto ₹28, Amul Taaza 500ml @BigBasket ₹30]
    - Output: Grouped as same product, best deal: Zepto ₹28
    """
    matcher = get_product_matcher()
    start_time = time.monotonic()
    
    # Convert Pydantic models to dicts (prefer platform-wise input if provided)
    platform_results = request.platform_results or {}
    if platform_results:
        limited_platform_results = {
            platform: items[:MAX_PLATFORM_ITEMS]
            for platform, items in platform_results.items()
        }
        products = [
            p.model_dump()
            for platform_products in limited_platform_results.values()
            for p in platform_products
        ]
    else:
        products = [p.model_dump() for p in request.products]
    
    result = await matcher.match_products(products)
    match_ms = int((time.monotonic() - start_time) * 1000)
    
    # Convert ProductGroup objects to dicts
    groups = []
    for group in result.product_groups:
        groups.append({
            "canonical_name": group.canonical_name,
            "brand": group.brand,
            "quantity": group.quantity,
            "products": group.products,
            "best_deal": group.best_deal,
            "price_range": group.price_range,
            "savings": group.savings
        })
    
    return {
        "ai_powered": result.ai_powered,
        "ai_meta": result.ai_meta,
        "product_groups": groups,
        "unmatched_products": result.unmatched_products,
        "timing_ms": {
            "total": match_ms,
            "match": match_ms
        },
        "stats": {
            "total_products": result.total_products,
            "total_groups": result.total_groups,
            "total_matched": result.total_matched,
            "total_unmatched": len(result.unmatched_products)
        }
    }


@app.get("/api/understand-query")
async def understand_query(q: str = Query(..., description="Search query to analyze")):
    """
    🆕 Understand search query intent using Gemini AI.
    
    Analyzes natural language query to extract:
    - Product type
    - Quantity
    - Brand
    - Category
    - Terms to include/exclude
    """
    gemini = get_gemini_service()
    
    result = await gemini.understand_query(q)
    
    return {
        "query": q,
        "ai_powered": result.get("ai_powered", False),
        "understanding": result
    }


@app.post("/api/smart-search-and-match")
async def smart_search_and_match(request: SmartSearchRequest):
    """
    🆕 Combined smart search + product matching in one call.
    
    1. Filters products using AI to remove irrelevant items
    2. Groups remaining products by similarity
    3. Returns best deals per product group
    
    This is the recommended endpoint for Android app integration.
    """
    smart_search_service = get_smart_search()
    matcher = get_product_matcher()
    start_time = time.monotonic()
    
    # Convert Pydantic models to dicts (prefer platform-wise input if provided)
    platform_results = request.platform_results or {}
    if platform_results:
        products = [
            p.model_dump()
            for platform_products in platform_results.values()
            for p in platform_products
        ]
    else:
        products = [p.model_dump() for p in request.products]
    
    # Step 1: Smart search filtering
    filter_start = time.monotonic()
    search_result = await smart_search_service.search(
        query=request.query,
        products=products,
        strict_mode=request.strict_mode
    )
    filter_ms = int((time.monotonic() - filter_start) * 1000)
    
    # Step 2: Match products across platforms
    match_start = time.monotonic()
    match_result = await matcher.match_products(search_result.products)
    match_ms = int((time.monotonic() - match_start) * 1000)
    total_ms = int((time.monotonic() - start_time) * 1000)
    
    # Convert ProductGroup objects to dicts
    groups = []
    for group in match_result.product_groups:
        groups.append({
            "canonical_name": group.canonical_name,
            "brand": group.brand,
            "quantity": group.quantity,
            "products": group.products,
            "best_deal": group.best_deal,
            "price_range": group.price_range,
            "savings": group.savings
        })
    
    platform_counts = {
        platform: len(items) for platform, items in limited_platform_results.items()
    } if platform_results else {}

    return {
        "query": request.query,
        "pincode": request.pincode,
        "ai_powered": search_result.ai_powered or match_result.ai_powered,
        "ai_meta": {
            "filter": search_result.ai_meta,
            "match": match_result.ai_meta
        },
        "query_understanding": search_result.query_understanding,
        
        # Matched product groups (for comparison view)
        "product_groups": groups,
        
        # All relevant products (flat list)
        "all_products": search_result.products,
        
        # Best overall deal
        "best_deal": search_result.best_deal,
        
        # Filtered out products (for debugging/transparency)
        "filtered_out": search_result.filtered_out,
        
        # Stats
        "stats": {
            "input_products": len(products),
            "relevant_products": search_result.total_found,
            "filtered_products": search_result.total_filtered,
            "product_groups": match_result.total_groups,
            "matched_products": match_result.total_matched,
            "platform_counts": platform_counts
        },
        "timing_ms": {
            "total": total_ms,
            "filter": filter_ms,
            "match": match_ms
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    gemini = get_gemini_service()
    return {
        "status": "healthy",
        "version": "2.0.0",
        "ai_available": gemini.is_available()
    }


@app.get("/api/gemini-ping")
async def gemini_ping():
    """Quick connectivity test for Gemini API."""
    gemini = get_gemini_service()
    return await gemini.ping()


@app.get("/api/gemini-models")
async def gemini_models():
    """List available Gemini models for this API key."""
    gemini = get_gemini_service()
    return await gemini.list_models()


@app.get("/api/groq-ping")
async def groq_ping():
    """Quick connectivity test for Groq (no fallback)."""
    gemini = get_gemini_service()
    return await gemini.ping_provider("groq")


@app.get("/api/quota-stats")
async def quota_stats():
    """Get current AI quota statistics for all providers."""
    ai_service = get_gemini_service()
    return ai_service.get_quota_stats()


@app.post("/api/reset-quota")
async def reset_quota():
    """Force reset AI quota tracking (for testing/debugging)."""
    ai_service = get_gemini_service()
    return ai_service.force_reset_quota()


@app.get("/api/ping-provider/{provider}")
async def ping_provider(provider: str):
    """Test connectivity to a specific AI provider."""
    ai_service = get_gemini_service()
    return await ai_service.ping_provider(provider)


class SmartSearchWithProviderRequest(BaseModel):
    """Request for smart search with specific provider/model"""
    query: str
    products: List[ProductInput] = []
    pincode: Optional[str] = "560001"
    strict_mode: bool = True
    provider: Optional[str] = None  # groq, gemini, mistral, etc.
    model: Optional[str] = None  # specific model override
    platform_results: Optional[Dict[str, List[ProductInput]]] = None


@app.post("/api/smart-search-with-provider")
async def smart_search_with_provider(request: SmartSearchWithProviderRequest):
    """
    AI-powered smart search with explicit provider/model selection.
    
    Use this to test specific AI models and compare their performance.
    No fallback - if the specified provider fails, returns error.
    
    Available providers: gemini, groq, mistral, cerebras, together, openrouter
    
    Example models:
    - gemini: gemini-2.5-flash, gemini-2.5-flash-lite, gemma-3-27b-it
    - groq: llama-3.3-70b-versatile, mixtral-8x7b-32768
    - mistral: mistral-small-latest
    """
    ai_service = get_gemini_service()
    start_time = time.monotonic()
    
    # Convert products
    platform_results = request.platform_results or {}
    if platform_results:
        limited_platform_results = {
            platform: items[:MAX_PLATFORM_ITEMS]
            for platform, items in platform_results.items()
        }
        products = [
            p.model_dump()
            for platform_products in limited_platform_results.values()
            for p in platform_products
        ]
    else:
        products = [p.model_dump() for p in request.products]
    
    filter_start = time.monotonic()
    result = await ai_service.filter_relevant_products_with_provider(
        query=request.query,
        products=products,
        strict_mode=request.strict_mode,
        provider=request.provider,
        model=request.model
    )
    filter_ms = int((time.monotonic() - filter_start) * 1000)
    total_ms = int((time.monotonic() - start_time) * 1000)
    
    # Find best deal from relevant products
    relevant_products = result.get("relevant_products", [])
    best_deal = None
    if relevant_products:
        available = [p for p in relevant_products if p.get("available", True) and p.get("price", 0) > 0]
        high_relevance = [p for p in available if p.get("relevance_score", 0) >= 70]
        candidates = high_relevance if high_relevance else available
        if candidates:
            best = min(candidates, key=lambda x: x.get("price", float("inf")))
            best_deal = {
                "name": best.get("name"),
                "price": best.get("price"),
                "platform": best.get("platform"),
                "relevance_score": best.get("relevance_score", 50)
            }
    
    platform_counts = {
        platform: len(items) for platform, items in limited_platform_results.items()
    } if platform_results else {}

    return {
        "query": request.query,
        "pincode": request.pincode,
        "provider": request.provider,
        "model": request.model,
        "ai_powered": result.get("ai_powered", False),
        "ai_meta": result.get("ai_meta"),
        "query_understanding": result.get("query_understanding", {}),
        "relevant_products": relevant_products,
        "filtered_out": result.get("filtered_out", []),
        "best_deal": best_deal,
        "error": result.get("error"),
        "timing_ms": {
            "total": total_ms,
            "filter": filter_ms
        },
        "stats": {
            "total_input": len(products),
            "total_relevant": len(relevant_products),
            "total_filtered": len(result.get("filtered_out", [])),
            "platform_counts": platform_counts
        }
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Starting PriceHunt API server v2.0...")
    print("🤖 AI-powered smart search enabled!")
    print(f"📱 Running on port: {port}")
    print(f"📚 API docs available at: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

