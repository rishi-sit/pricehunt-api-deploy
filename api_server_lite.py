"""
PriceHunt API Lite - AI-powered filtering AND fallback scraping
Designed for free tier hosting (Railway, Render, etc.)

This version:
1. Receives scraped products from Android app
2. Uses Gemini AI for smart filtering and product matching
3. Provides AI-powered HTML extraction as FALLBACK when client scraping fails
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import os
import time

# AI-powered modules
from app.smart_search import get_smart_search
from app.product_matcher import get_product_matcher
from app.gemini_service import get_gemini_service
from app.ai_scraper import get_ai_scraper

MAX_PLATFORM_ITEMS = 10

app = FastAPI(
    title="PriceHunt API Lite",
    description="AI-powered product filtering and matching API (no scraping)",
    version="2.0.0-lite"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI services
smart_search = get_smart_search()
product_matcher = get_product_matcher()
gemini = get_gemini_service()
ai_scraper = get_ai_scraper()


# ============== Pydantic Models ==============

class ProductInput(BaseModel):
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
    query: str
    products: List[ProductInput] = []
    pincode: Optional[str] = "560001"
    strict_mode: bool = True
    platform_results: Optional[Dict[str, List[ProductInput]]] = None


class MatchProductsRequest(BaseModel):
    products: List[ProductInput]


class AIExtractRequest(BaseModel):
    """Request for AI-powered HTML extraction (fallback scraping)"""
    html: str
    platform: str
    search_query: str
    base_url: str


class MultiPlatformExtractRequest(BaseModel):
    """Request for extracting from multiple platform HTMLs"""
    platforms: List[Dict[str, str]]  # [{"platform": str, "html": str, "base_url": str}]
    search_query: str


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """API info"""
    return {
        "name": "PriceHunt API Lite",
        "version": "2.1.0-lite",
        "description": "AI-powered filtering + fallback scraping",
        "ai_enabled": gemini.is_available(),
        "ai_scraper_enabled": ai_scraper.is_available(),
        "endpoints": {
            "smart_search": "POST /api/smart-search",
            "match_products": "POST /api/match-products",
            "combined": "POST /api/smart-search-and-match",
            "understand_query": "GET /api/understand-query",
            "ai_extract": "POST /api/ai-extract (NEW - fallback scraping)",
            "ai_extract_multi": "POST /api/ai-extract-multi (NEW - multi-platform)",
            "health": "GET /api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check with AI status"""
    return {
        "status": "healthy",
        "version": "2.1.0-lite",
        "ai_available": gemini.is_available(),
        "ai_scraper_available": ai_scraper.is_available(),
        "mode": "lite (with AI fallback scraping)"
    }


@app.get("/api/gemini-ping")
async def gemini_ping():
    """Quick connectivity test for Gemini API."""
    return await gemini.ping()


@app.post("/api/smart-search")
async def smart_search_endpoint(request: SmartSearchRequest):
    """
    Filter products using AI to keep only relevant results.
    Example: "milk" query filters out "milkshake", "milkmade", etc.
    """
    try:
        start_time = time.monotonic()
        platform_results = request.platform_results or {}
        if platform_results:
            limited_platform_results = {
                platform: items[:MAX_PLATFORM_ITEMS]
                for platform, items in platform_results.items()
            }
            products_dict = [
                p.model_dump()
                for platform_products in limited_platform_results.values()
                for p in platform_products
            ]
        else:
            products_dict = [p.model_dump() for p in request.products]

        non_empty_platforms = (
            sum(1 for items in limited_platform_results.values() if items)
            if platform_results
            else 0
        )
        use_ai = not platform_results or non_empty_platforms > 1
        ai_skip_reason = None if use_ai else "single_platform"

        filter_start = time.monotonic()
        result = await smart_search.search(
            query=request.query,
            products=products_dict,
            strict_mode=request.strict_mode,
            use_ai=use_ai,
            ai_skip_reason=ai_skip_reason
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
                "total_input": len(products_dict),
                "total_relevant": result.total_found,
                "total_filtered": result.total_filtered,
                "platform_counts": platform_counts
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/match-products")
async def match_products_endpoint(request: MatchProductsRequest):
    """
    Match similar products across platforms.
    Groups products like "Amul Milk 500ml" from different platforms.
    """
    try:
        start_time = time.monotonic()
        products_dict = [p.model_dump() for p in request.products]

        result = await product_matcher.match_products(products_dict)
        match_ms = int((time.monotonic() - start_time) * 1000)

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/smart-search-and-match")
async def smart_search_and_match(request: SmartSearchRequest):
    """
    Combined endpoint: Filter irrelevant products AND match similar ones.
    Recommended for Android app integration.
    """
    try:
        start_time = time.monotonic()
        platform_results = request.platform_results or {}
        if platform_results:
            limited_platform_results = {
                platform: items[:MAX_PLATFORM_ITEMS]
                for platform, items in platform_results.items()
            }
            products_dict = [
                p.model_dump()
                for platform_products in limited_platform_results.values()
                for p in platform_products
            ]
        else:
            products_dict = [p.model_dump() for p in request.products]
        
        # Step 1: Smart filter
        filter_start = time.monotonic()
        filter_result = await smart_search.search(
            query=request.query,
            products=products_dict,
            strict_mode=request.strict_mode
        )
        filter_ms = int((time.monotonic() - filter_start) * 1000)

        filtered_products = filter_result.products

        # Step 2: Match similar products
        match_start = time.monotonic()
        match_result = await product_matcher.match_products(filtered_products)
        match_ms = int((time.monotonic() - match_start) * 1000)
        total_ms = int((time.monotonic() - start_time) * 1000)

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
            "ai_powered": filter_result.ai_powered or match_result.ai_powered,
            "ai_meta": {
                "filter": filter_result.ai_meta,
                "match": match_result.ai_meta
            },
            "query_understanding": filter_result.query_understanding,
            "product_groups": groups,
            "all_products": filter_result.products,
            "best_deal": filter_result.best_deal,
            "filtered_out": filter_result.filtered_out,
            "stats": {
                "input_products": len(products_dict),
                "relevant_products": filter_result.total_found,
                "filtered_products": filter_result.total_filtered,
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/understand-query")
async def understand_query(q: str):
    """
    Use AI to understand search intent.
    Returns: product type, brand hints, quantity preferences, etc.
    """
    try:
        understanding = await gemini.understand_query(q)
        return {
            "query": q,
            "understanding": understanding,
            "ai_powered": gemini.is_available()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== AI Fallback Scraping Endpoints ==============

@app.post("/api/ai-extract")
async def ai_extract_products(request: AIExtractRequest):
    """
    AI-powered product extraction from raw HTML.
    
    USE THIS WHEN CLIENT-SIDE SCRAPING FAILS.
    
    The Android app should:
    1. Try normal WebView scraping first
    2. If extraction returns 0 products, send the raw HTML here
    3. Gemini AI will extract products intelligently
    
    This handles:
    - Anti-bot protected pages
    - JavaScript-rendered content that WebView captured
    - Pages with unusual HTML structures
    - New/changed website layouts
    """
    try:
        if not ai_scraper.is_available():
            raise HTTPException(
                status_code=503, 
                detail="AI scraper not available - GEMINI_API_KEY not set"
            )
        
        result = await ai_scraper.extract_products_from_html(
            html=request.html,
            platform=request.platform,
            search_query=request.search_query,
            base_url=request.base_url
        )
        
        return {
            "platform": request.platform,
            "search_query": request.search_query,
            "products": result.get("products", []),
            "products_found": len(result.get("products", [])),
            "extraction_method": result.get("extraction_method", "none"),
            "confidence": result.get("confidence", 0),
            "ai_powered": result.get("ai_powered", False),
            "error": result.get("error")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai-extract-multi")
async def ai_extract_multi_platform(request: MultiPlatformExtractRequest):
    """
    Extract products from multiple platform HTMLs in parallel.
    
    Send all failed platform HTMLs at once for efficient processing.
    
    Request format:
    {
        "platforms": [
            {"platform": "Zepto", "html": "...", "base_url": "https://zeptonow.com"},
            {"platform": "Blinkit", "html": "...", "base_url": "https://blinkit.com"}
        ],
        "search_query": "milk"
    }
    """
    try:
        if not ai_scraper.is_available():
            raise HTTPException(
                status_code=503,
                detail="AI scraper not available - GEMINI_API_KEY not set"
            )
        
        result = await ai_scraper.extract_from_multiple_platforms(
            platform_html_list=request.platforms,
            search_query=request.search_query
        )
        
        return {
            "search_query": request.search_query,
            "results": result.get("results", {}),
            "total_products": result.get("total_products", 0),
            "ai_powered": result.get("ai_powered", False)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/smart-extract-and-filter")
async def smart_extract_and_filter(request: MultiPlatformExtractRequest):
    """
    COMPLETE PIPELINE: Extract + Filter + Match
    
    1. Extract products from all provided HTMLs using AI
    2. Filter to keep only relevant products
    3. Match similar products across platforms
    4. Return best deals
    
    This is the ULTIMATE FALLBACK - handles everything server-side.
    """
    try:
        if not ai_scraper.is_available():
            raise HTTPException(
                status_code=503,
                detail="AI scraper not available"
            )
        
        # Step 1: Extract from all platforms
        extraction_result = await ai_scraper.extract_from_multiple_platforms(
            platform_html_list=request.platforms,
            search_query=request.search_query
        )
        
        # Collect all extracted products
        all_products = []
        for platform, result in extraction_result.get("results", {}).items():
            for product in result.get("products", []):
                all_products.append(product)
        
        if not all_products:
            return {
                "search_query": request.search_query,
                "extraction_results": extraction_result.get("results", {}),
                "filtered_products": [],
                "product_groups": [],
                "best_deal": None,
                "total_products": 0,
                "ai_powered": True
            }
        
        # Step 2: Filter relevant products
        filter_result = await smart_search.search(
            query=request.search_query,
            products=all_products
        )
        
        filtered_products = filter_result.get("products", all_products)
        
        # Step 3: Match similar products
        match_result = await product_matcher.match_products(filtered_products)
        
        return {
            "search_query": request.search_query,
            "stats": {
                "platforms_processed": len(request.platforms),
                "total_extracted": len(all_products),
                "after_filtering": len(filtered_products),
                "product_groups": len(match_result.get("groups", []))
            },
            "extraction_results": extraction_result.get("results", {}),
            "filtered_products": filtered_products,
            "product_groups": match_result.get("groups", []),
            "best_deal": filter_result.get("best_deal"),
            "ai_powered": True
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Starting PriceHunt API Lite v2.0...")
    print("🤖 AI-powered filtering (no scraping)")
    print(f"📱 Running on port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
