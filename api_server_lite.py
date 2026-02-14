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
from app.ai_service import get_ai_service as get_gemini_service
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
    device_id: Optional[str] = None  # For analytics tracking


class MultiPlatformExtractRequest(BaseModel):
    """Request for extracting from multiple platform HTMLs"""
    platforms: List[Dict[str, str]]  # [{"platform": str, "html": str, "base_url": str}]
    search_query: str
    device_id: Optional[str] = None  # For analytics tracking


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
            "groq_ping": "GET /api/groq-ping",
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


@app.post("/api/reset-quota")
async def reset_quota():
    """
    Force reset all AI quota tracking.
    Use this when all providers are exhausted and you need to test immediately.
    WARNING: Use sparingly - may cause actual rate limit errors if called too often.
    """
    ai_service_result = gemini.force_reset_quota()
    ai_scraper_result = ai_scraper.force_reset_quota()
    
    return {
        "message": "All AI quotas reset successfully",
        "ai_service": ai_service_result,
        "ai_scraper": ai_scraper_result,
        "ai_available": gemini.is_available(),
        "ai_scraper_available": ai_scraper.is_available()
    }


@app.get("/api/gemini-ping")
async def gemini_ping():
    """Quick connectivity test for Gemini API."""
    return await gemini.ping()


@app.get("/api/gemini-models")
async def gemini_models():
    """List available Gemini models for this API key."""
    return await gemini.list_models()


@app.get("/api/ai-accuracy")
async def ai_accuracy(provider: str = "groq", model: Optional[str] = None):
    """
    Simple accuracy check for AI filtering on a small built-in dataset.
    Intended for quick sanity checks, not a full benchmark.
    """
    tests = [
        {
            "name": "milk_basic",
            "query": "milk",
            "products": [
                {"name": "Amul Toned Milk 500ml", "price": 28, "platform": "Zepto"},
                {"name": "Milkmaid Condensed 400g", "price": 99, "platform": "BigBasket"},
                {"name": "Cadbury Dairy Milk Chocolate", "price": 50, "platform": "Amazon"},
                {"name": "Mother Dairy Full Cream Milk 1L", "price": 68, "platform": "JioMart"},
                {"name": "Nestle Milkshake Strawberry", "price": 35, "platform": "Blinkit"}
            ],
            "expected_relevant": [
                "Amul Toned Milk 500ml",
                "Mother Dairy Full Cream Milk 1L"
            ]
        },
        {
            "name": "milk_derived",
            "query": "milk",
            "products": [
                {"name": "Nandini Toned Milk 500ml", "price": 27, "platform": "Zepto"},
                {"name": "Nandini Curd 400g", "price": 35, "platform": "Blinkit"},
                {"name": "Milk Powder 1kg", "price": 380, "platform": "Amazon"},
                {"name": "Fresh Cow Milk 1L", "price": 60, "platform": "BigBasket"},
                {"name": "Chocolate Milk Drink 200ml", "price": 25, "platform": "Instamart"}
            ],
            "expected_relevant": [
                "Nandini Toned Milk 500ml",
                "Fresh Cow Milk 1L"
            ]
        },
        {
            "name": "bread_basic",
            "query": "bread",
            "products": [
                {"name": "Britannia Bread 400g", "price": 40, "platform": "Blinkit"},
                {"name": "Whole Wheat Bread 400g", "price": 45, "platform": "BigBasket"},
                {"name": "Bread Knife 8 inch", "price": 120, "platform": "Amazon"},
                {"name": "Toaster 2 Slice", "price": 999, "platform": "Amazon"},
                {"name": "Wheat Flour 1kg", "price": 55, "platform": "Zepto"}
            ],
            "expected_relevant": [
                "Britannia Bread 400g",
                "Whole Wheat Bread 400g"
            ]
        },
        {
            "name": "eggs_basic",
            "query": "eggs",
            "products": [
                {"name": "Farm Fresh Eggs 12pcs", "price": 90, "platform": "BigBasket"},
                {"name": "Eggs 6pcs", "price": 48, "platform": "Zepto"},
                {"name": "Eggless Mayo 250g", "price": 60, "platform": "Blinkit"},
                {"name": "Eggless Cake Mix 500g", "price": 80, "platform": "Amazon"},
                {"name": "Chicken Breast 500g", "price": 220, "platform": "Instamart"}
            ],
            "expected_relevant": [
                "Farm Fresh Eggs 12pcs",
                "Eggs 6pcs"
            ]
        },
        {
            "name": "rice_5kg",
            "query": "rice 5kg",
            "products": [
                {"name": "India Gate Basmati Rice 5kg", "price": 450, "platform": "Amazon"},
                {"name": "Sona Masoori Rice 5kg", "price": 399, "platform": "BigBasket"},
                {"name": "Rice Flour 1kg", "price": 60, "platform": "Blinkit"},
                {"name": "Brown Rice 1kg", "price": 120, "platform": "Zepto"},
                {"name": "Poha 1kg", "price": 70, "platform": "Instamart"}
            ],
            "expected_relevant": [
                "India Gate Basmati Rice 5kg",
                "Sona Masoori Rice 5kg"
            ]
        },
        {
            "name": "oil_1l",
            "query": "sunflower oil 1L",
            "products": [
                {"name": "Fortune Sunflower Oil 1L", "price": 170, "platform": "Amazon"},
                {"name": "Saffola Sunflower Oil 1L", "price": 180, "platform": "BigBasket"},
                {"name": "Mustard Oil 1L", "price": 140, "platform": "JioMart"},
                {"name": "Sunflower Oil 5L", "price": 780, "platform": "Amazon"},
                {"name": "Olive Oil 500ml", "price": 350, "platform": "Blinkit"}
            ],
            "expected_relevant": [
                "Fortune Sunflower Oil 1L",
                "Saffola Sunflower Oil 1L"
            ]
        },
        {
            "name": "apple_basic",
            "query": "apple",
            "products": [
                {"name": "Fresh Apple 1kg", "price": 120, "platform": "BigBasket"},
                {"name": "Red Apple 4pcs", "price": 80, "platform": "Zepto"},
                {"name": "Apple Juice 1L", "price": 110, "platform": "Blinkit"},
                {"name": "Apple Cider Vinegar 500ml", "price": 180, "platform": "Amazon"},
                {"name": "Pineapple 1kg", "price": 90, "platform": "Instamart"}
            ],
            "expected_relevant": [
                "Fresh Apple 1kg",
                "Red Apple 4pcs"
            ]
        },
        {
            "name": "banana_fresh",
            "query": "banana",
            "products": [
                {"name": "Banana Robusta 1kg", "price": 60, "platform": "Zepto"},
                {"name": "Yellaki Banana 12 pcs", "price": 70, "platform": "BigBasket"},
                {"name": "Banana Chips 150g", "price": 45, "platform": "Amazon"},
                {"name": "Banana Cake Slice", "price": 55, "platform": "Blinkit"},
                {"name": "Plantain Chips 200g", "price": 65, "platform": "Instamart"}
            ],
            "expected_relevant": [
                "Banana Robusta 1kg",
                "Yellaki Banana 12 pcs"
            ]
        }
    ]

    results = []
    total_items = 0
    total_tp = total_fp = total_fn = total_tn = 0

    for test in tests:
        ai_result = await gemini.filter_relevant_products_with_provider(
            query=test["query"],
            products=test["products"],
            strict_mode=True,
            provider=provider,
            model=model
        )
        predicted = {p.get("name", "").strip().lower() for p in ai_result.get("relevant_products", [])}
        expected = {name.strip().lower() for name in test["expected_relevant"]}
        all_items = {p.get("name", "").strip().lower() for p in test["products"]}

        tp = len(predicted & expected)
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        tn = len((all_items - predicted) & (all_items - expected))

        total = len(all_items)
        total_items += total
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        results.append({
            "name": test["name"],
            "provider": (ai_result.get("ai_meta") or {}).get("provider"),
            "model": (ai_result.get("ai_meta") or {}).get("model"),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "predicted_relevant": sorted(predicted)
        })

    overall_accuracy = (total_tp + total_tn) / total_items if total_items else 0.0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0

    return {
        "provider_requested": provider,
        "model_requested": model,
        "tests": results,
        "overall": {
            "accuracy": overall_accuracy,
            "precision": overall_precision,
            "recall": overall_recall,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
            "total_items": total_items
        }
    }


@app.get("/api/groq-ping")
async def groq_ping():
    """Quick connectivity test for Groq (no fallback)."""
    return await gemini.ping_provider("groq")


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

        filter_start = time.monotonic()
        result = await smart_search.search(
            query=request.query,
            products=products_dict,
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
    AI-powered product extraction from raw HTML with relevance filtering.
    
    PRIMARY METHOD: All scraping is now AI-based.
    
    This endpoint:
    1. Extracts products from HTML using AI
    2. Filters products by relevance to search query (AI-powered)
    3. Returns only relevant products with quantity and price_per_unit
    
    This handles:
    - Anti-bot protected pages
    - JavaScript-rendered content
    - Pages with unusual HTML structures
    - New/changed website layouts
    - Dynamic SPAs (Instamart, JioMart, etc.)
    """
    try:
        if not ai_scraper.is_available():
            raise HTTPException(
                status_code=503, 
                detail="AI scraper not available - GEMINI_API_KEY not set"
            )
        
        # Step 1: Extract products from HTML
        result = await ai_scraper.extract_products_from_html(
            html=request.html,
            platform=request.platform,
            search_query=request.search_query,
            base_url=request.base_url,
            device_id=request.device_id
        )
        
        extracted_products = result.get("products", [])
        if not extracted_products:
            return {
                "platform": request.platform,
                "search_query": request.search_query,
                "products": [],
                "products_found": 0,
                "products_after_filtering": 0,
                "extraction_method": result.get("extraction_method", "none"),
                "confidence": result.get("confidence", 0),
                "ai_powered": result.get("ai_powered", False),
                "filtered": True,
                "error": result.get("error")
            }
        
        # Step 2: Filter products by relevance using AI
        ai_service = get_gemini_service()
        filter_result = await ai_service.filter_relevant_products(
            query=request.search_query,
            products=extracted_products,
            strict_mode=True
        )
        
        relevant_products = filter_result.get("relevant_products", [])
        filtered_count = len(filter_result.get("filtered_out", []))
        
        return {
            "platform": request.platform,
            "search_query": request.search_query,
            "products": relevant_products,
            "products_found": len(extracted_products),
            "products_after_filtering": len(relevant_products),
            "filtered_out": filtered_count,
            "extraction_method": result.get("extraction_method", "none"),
            "confidence": result.get("confidence", 0),
            "ai_powered": True,
            "filtered": filter_result.get("ai_powered", False),
            "ai_meta": filter_result.get("ai_meta", {}),
            "error": result.get("error")
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ /api/ai-extract error for {request.platform}: {error_detail}")
        # Return a proper response instead of raising 500, so client can handle it
        return {
            "platform": request.platform,
            "search_query": request.search_query,
            "products": [],
            "products_found": 0,
            "products_after_filtering": 0,
            "filtered_out": 0,
            "extraction_method": "error",
            "confidence": 0,
            "ai_powered": False,
            "filtered": False,
            "error": f"Backend error: {str(e)}"
        }


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
            search_query=request.search_query,
            device_id=request.device_id
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
            search_query=request.search_query,
            device_id=request.device_id
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


# ============== SERVER-SIDE FULL SEARCH ==============

@app.get("/api/full-search")
async def full_search(
    q: str = "",
    pincode: str = "560001",
    strict: bool = True
):
    """
    FULL SERVER-SIDE SEARCH: Scrape + AI filter + match in one call.
    
    This is the RECOMMENDED endpoint for the Android app.
    The server handles ALL scraping (via Playwright/HTTP) so the app
    doesn't need any scraping logic. Benefits:
    - Scraping can be fixed server-side without app updates
    - Playwright browser is more reliable than Android WebView
    - AI filtering + matching happens in the same request
    
    Usage: GET /api/full-search?q=banana&pincode=560001
    """
    import asyncio as _asyncio

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    query = q.strip()
    start_time = time.monotonic()

    # Import scrapers lazily (only when this endpoint is called)
    try:
        from app.scrapers.amazon import AmazonScraper
        from app.scrapers.amazon_fresh import AmazonFreshScraper
        from app.scrapers.flipkart import FlipkartScraper
        from app.scrapers.bigbasket import BigBasketScraper
        from app.scrapers.zepto import ZeptoScraper
        from app.scrapers.blinkit import BlinkitScraper
        from app.scrapers.instamart import InstamartScraper
        from app.scrapers.jiomart import JioMartScraper

        scrapers = {
            "Amazon": AmazonScraper(pincode),
            "Amazon Fresh": AmazonFreshScraper(pincode),
            "Flipkart": FlipkartScraper(pincode),
            "BigBasket": BigBasketScraper(pincode),
            "Zepto": ZeptoScraper(pincode),
            "Blinkit": BlinkitScraper(pincode),
            "Instamart": InstamartScraper(pincode),
            "JioMart": JioMartScraper(pincode),
        }
    except ImportError as e:
        # Playwright not available on this deployment — return error
        raise HTTPException(
            status_code=503,
            detail=f"Server-side scraping not available: {e}. Use POST /api/smart-search instead."
        )

    # Scrape all platforms in parallel with per-platform timeout
    async def scrape_one(name, scraper):
        try:
            products = await _asyncio.wait_for(scraper.safe_search(query), timeout=25.0)
            print(f"  {name}: {len(products)} products")
            return name, [
                {
                    "name": p.name,
                    "price": p.price,
                    "original_price": p.original_price,
                    "discount": p.discount,
                    "platform": p.platform,
                    "url": p.url,
                    "image_url": p.image_url,
                    "rating": p.rating,
                    "delivery_time": p.delivery_time or "",
                    "available": p.available,
                }
                for p in products
            ]
        except Exception as e:
            print(f"  {name}: Error - {e}")
            return name, []

    print(f"🔍 FULL SEARCH: '{query}' (pincode={pincode})")
    tasks = [scrape_one(name, scraper) for name, scraper in scrapers.items()]
    platform_results_list = await _asyncio.gather(*tasks)

    scrape_ms = int((time.monotonic() - start_time) * 1000)

    # Build platform results dict
    platform_results = {}
    all_products = []
    for name, products in platform_results_list:
        if products:
            platform_results[name] = products
            all_products.extend(products)

    print(f"📊 Scraped {len(all_products)} products from {len(platform_results)} platforms in {scrape_ms}ms")

    # If no products scraped, return empty
    if not all_products:
        return {
            "query": query,
            "pincode": pincode,
            "ai_powered": False,
            "results": [],
            "platform_results": {},
            "best_deal": None,
            "filtered_out": [],
            "product_groups": [],
            "stats": {
                "total_scraped": 0,
                "platforms_with_results": 0,
                "total_relevant": 0,
                "total_filtered": 0,
            },
            "timing_ms": {"scrape": scrape_ms, "total": scrape_ms},
        }

    # AI filter + match using existing smart search
    filter_start = time.monotonic()
    try:
        result = await smart_search.search(
            query=query,
            products=all_products,
            strict_mode=strict,
            use_ai=True,
        )
        filter_ms = int((time.monotonic() - filter_start) * 1000)

        # Group results by platform for display
        results_by_platform = {}
        for p in result.products:
            plat = p.get("platform", "Unknown")
            results_by_platform.setdefault(plat, []).append(p)

        total_ms = int((time.monotonic() - start_time) * 1000)

        return {
            "query": query,
            "pincode": pincode,
            "ai_powered": result.ai_powered,
            "ai_meta": result.ai_meta,
            "results": result.products,
            "platform_results": results_by_platform,
            "best_deal": result.best_deal,
            "filtered_out": result.filtered_out,
            "product_groups": [],
            "stats": {
                "total_scraped": len(all_products),
                "platforms_with_results": len(platform_results),
                "total_relevant": result.total_found,
                "total_filtered": result.total_filtered,
            },
            "timing_ms": {
                "scrape": scrape_ms,
                "ai_filter": filter_ms,
                "total": total_ms,
            },
        }
    except Exception as e:
        total_ms = int((time.monotonic() - start_time) * 1000)
        print(f"❌ AI filtering failed: {e}, returning raw results")
        return {
            "query": query,
            "pincode": pincode,
            "ai_powered": False,
            "results": all_products,
            "platform_results": platform_results,
            "best_deal": min(all_products, key=lambda x: x["price"]) if all_products else None,
            "filtered_out": [],
            "product_groups": [],
            "stats": {
                "total_scraped": len(all_products),
                "platforms_with_results": len(platform_results),
                "total_relevant": len(all_products),
                "total_filtered": 0,
            },
            "timing_ms": {"scrape": scrape_ms, "total": total_ms},
            "error": str(e),
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Starting PriceHunt API Lite v2.0...")
    print("🤖 AI-powered filtering + optional server-side scraping")
    print(f"📱 Running on port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


# ============================================================================
# Analytics Dashboard Endpoints
# ============================================================================

from app.analytics import (
    ScrapeLogRequest, BulkLogRequest, DashboardQueryRequest,
    AIProcessingLogRequest,
    log_scrape_event, log_bulk_events, get_dashboard_data,
    get_recent_logs, get_all_devices, log_ai_processing,
    get_ai_processing_stats, get_combined_dashboard
)


@app.post("/api/analytics/log")
async def log_analytics(log: ScrapeLogRequest):
    """
    Log a single scrape event from the Android app.
    
    Called after each platform scrape with metrics:
    - device_id: Unique device identifier
    - platform: Platform name (Zepto, BigBasket, etc.)
    - scrape_source: 'device', 'ai_fallback', 'playwright', 'cache'
    - html_size_kb: Size of HTML response
    - products_scraped: Number of products found
    - relevant_products: Number of relevant products after AI filtering
    - ai_model: Model used for extraction (groq-mistral, gemini, etc.)
    """
    try:
        log_id = log_scrape_event(log)
        return {
            "success": True,
            "log_id": log_id,
            "message": "Analytics logged successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/analytics/log-bulk")
async def log_analytics_bulk(request: BulkLogRequest):
    """
    Log multiple scrape events in a batch (more efficient).
    
    Use this to send all platform results at once after a search completes.
    """
    try:
        log_ids = log_bulk_events(request.logs)
        return {
            "success": True,
            "log_ids": log_ids,
            "count": len(log_ids),
            "message": f"Logged {len(log_ids)} analytics events"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/analytics/dashboard")
async def get_analytics_dashboard(request: DashboardQueryRequest):
    """
    Get dashboard data for a specific device and date range.
    
    Returns:
    - Platform-wise statistics
    - Scrape source breakdown (device vs server)
    - AI model usage
    - Success rates
    - Products scraped vs relevant products
    """
    try:
        dashboard = get_dashboard_data(
            device_id=request.device_id,
            start_date=request.start_date,
            end_date=request.end_date
        )
        return {
            "success": True,
            "data": dashboard.model_dump()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/analytics/dashboard/{device_id}")
async def get_analytics_dashboard_simple(
    device_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    GET endpoint for dashboard - easier to test in browser.
    """
    try:
        dashboard = get_dashboard_data(
            device_id=device_id,
            start_date=start_date,
            end_date=end_date
        )
        return {
            "success": True,
            "data": dashboard.model_dump()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/analytics/logs/{device_id}")
async def get_device_logs(
    device_id: str,
    limit: int = 100,
    platform: Optional[str] = None
):
    """
    Get recent scrape logs for a device.
    Useful for debugging specific platform issues.
    """
    try:
        logs = get_recent_logs(device_id, limit, platform)
        return {
            "success": True,
            "logs": [log.model_dump() for log in logs],
            "count": len(logs)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/analytics/devices")
async def list_all_devices():
    """
    List all devices that have sent analytics.
    Useful for admin dashboard.
    """
    try:
        devices = get_all_devices()
        return {
            "success": True,
            "devices": devices,
            "count": len(devices)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/analytics/ai-processing")
async def log_ai_processing_event(log: AIProcessingLogRequest):
    """
    Log AI processing event from backend.
    
    Called internally when AI service processes HTML for product extraction.
    Tracks:
    - ai_provider: Provider used (groq, gemini, mistral)
    - ai_model: Model used (mixtral-8x7b, gemini-flash, etc.)
    - input_html_size_kb: Size of input HTML
    - products_found: Products extracted
    - products_filtered: Relevant products after filtering
    - latency_ms: Processing time
    - fallback_reason: Why fallback was used (if applicable)
    """
    try:
        log_id = log_ai_processing(log)
        return {
            "success": True,
            "log_id": log_id,
            "message": "AI processing logged successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/analytics/ai-stats/{device_id}")
async def get_ai_stats(
    device_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get AI processing statistics for a device.
    
    Returns:
    - Total AI requests
    - Success/failure counts
    - Average latency
    - Provider breakdown (Groq vs Gemini vs Mistral)
    - Fallback reasons
    """
    try:
        stats = get_ai_processing_stats(device_id, start_date, end_date)
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/analytics/combined/{device_id}")
async def get_combined_analytics(
    device_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get combined analytics dashboard with both scrape logs and AI processing stats.
    
    This is the UNIFIED dashboard that combines:
    - Android scrape metrics (device vs server, HTML sizes, product counts)
    - Backend AI processing metrics (models used, latency, success rates)
    """
    try:
        combined = get_combined_dashboard(device_id, start_date, end_date)
        return {
            "success": True,
            "data": combined
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
