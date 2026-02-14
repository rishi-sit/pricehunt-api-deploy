"""FastAPI Price Comparator Application."""
import asyncio
import json
from typing import Optional, List, Dict, AsyncGenerator
from dataclasses import asdict
from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.scrapers import (
    AmazonScraper,
    AmazonFreshScraper,
    FlipkartScraper,
    FlipkartMinutesScraper,
    ZeptoScraper,
    InstamartScraper,
    BlinkitScraper,
    BigBasketScraper,
    JioMartQuickScraper,
    JioMartScraper,
)
from app.scrapers.base import ProductResult
from app.cache import cache

app = FastAPI(
    title="Price Comparator",
    description="Compare prices across Amazon, Flipkart, Zepto, Instamart, and Blinkit",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


class SearchRequest(BaseModel):
    """Search request model."""
    products: List[str]
    pincode: str = "560087"


class ProductComparison(BaseModel):
    """Product comparison result."""
    query: str
    results: List[Dict]
    lowest_price: Optional[Dict] = None
    total_platforms: int = 0


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/search")
async def search_single(
    q: str = Query(..., description="Product search query"),
    pincode: str = Query("560087", description="Delivery pincode")
):
    """Search for a single product across all platforms."""
    comparison = await compare_prices(q, pincode)
    return comparison


@app.get("/api/search/stream")
async def search_stream(
    q: str = Query(..., description="Product search query"),
    pincode: str = Query("560087", description="Delivery pincode")
):
    """Stream search results as they arrive from each platform using SSE."""
    return StreamingResponse(
        stream_search_results(q, pincode),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


async def stream_search_results(query: str, pincode: str) -> AsyncGenerator[str, None]:
    """Generator that yields SSE events as each scraper completes, with caching support."""
    
    # Initialize all scrapers with their configs
    scraper_configs = [
        ("Amazon Fresh", AmazonFreshScraper(pincode), 25.0),
        ("Flipkart Minutes", FlipkartMinutesScraper(pincode), 25.0),
        ("JioMart Quick", JioMartQuickScraper(pincode), 25.0),
        ("BigBasket", BigBasketScraper(pincode), 25.0),
        ("Amazon", AmazonScraper(pincode), 25.0),
        ("Flipkart", FlipkartScraper(pincode), 25.0),
        ("JioMart", JioMartScraper(pincode), 25.0),
        ("Zepto", ZeptoScraper(pincode), 40.0),
    ]
    
    # Send initial event with platform list
    platforms = [name for name, _, _ in scraper_configs]
    yield f"event: init\ndata: {json.dumps({'query': query, 'platforms': platforms})}\n\n"
    
    # Check cache for each platform and separate cached vs non-cached
    cached_results = []
    platforms_to_fetch = []
    
    for name, scraper, timeout in scraper_configs:
        cached_data, is_stale = cache.get(name, query, pincode)
        
        if cached_data is not None:
            # We have cached data - send it immediately
            cached_results.append((name, cached_data, is_stale))
            
            # If stale, also add to fetch list for background refresh
            if is_stale:
                platforms_to_fetch.append((name, scraper, timeout))
        else:
            # No cache - need to fetch
            platforms_to_fetch.append((name, scraper, timeout))
    
    # Send cached results immediately (super fast!)
    for name, results, is_stale in cached_results:
        event_data = {
            "platform": name,
            "results": results,
            "count": len(results),
            "cached": True,
            "stale": is_stale
        }
        yield f"event: platform\ndata: {json.dumps(event_data)}\n\n"
    
    # If all results were cached and fresh, we're done!
    if not platforms_to_fetch:
        yield f"event: complete\ndata: {json.dumps({'status': 'done', 'all_cached': True})}\n\n"
        return
    
    # Fetch fresh data for non-cached or stale platforms
    async def run_scraper(name: str, scraper, timeout: float):
        """Run a single scraper with timeout and cache the results."""
        try:
            results = await asyncio.wait_for(scraper.search(query), timeout=timeout)
            results_list = [asdict(r) for r in results] if results else []
            
            # Cache the fresh results
            cache.set(name, query, pincode, results_list)
            
            return name, results_list
        except asyncio.TimeoutError:
            print(f"{name}: TIMEOUT")
            return name, []
        except Exception as e:
            print(f"{name}: ERROR - {e}")
            return name, []
    
    # Create tasks for platforms that need fetching
    tasks = {
        asyncio.create_task(run_scraper(name, scraper, timeout)): name
        for name, scraper, timeout in platforms_to_fetch
    }
    
    # Track which platforms already had cached data sent (for stale-while-revalidate)
    cached_platform_names = {name for name, _, is_stale in cached_results if is_stale}
    
    # Yield fresh results as each scraper completes
    for completed_task in asyncio.as_completed(tasks.keys()):
        name, results = await completed_task
        
        # For stale-while-revalidate: only send if results are different or better
        # For non-cached: always send
        if name in cached_platform_names:
            # This was a background refresh - send update event
            event_data = {
                "platform": name,
                "results": results,
                "count": len(results),
                "cached": False,
                "refreshed": True  # Indicates this is a refresh of stale data
            }
            yield f"event: refresh\ndata: {json.dumps(event_data)}\n\n"
        else:
            # Fresh fetch - send normal platform event
            event_data = {
                "platform": name,
                "results": results,
                "count": len(results),
                "cached": False
            }
            yield f"event: platform\ndata: {json.dumps(event_data)}\n\n"
    
    # Send completion event
    yield f"event: complete\ndata: {json.dumps({'status': 'done', 'all_cached': False})}\n\n"


@app.post("/api/search/bulk")
async def search_bulk(request: SearchRequest):
    """Search for multiple products across all platforms."""
    results = []
    
    for product in request.products:
        if product.strip():
            comparison = await compare_prices(product.strip(), request.pincode)
            results.append(comparison)
    
    return {"comparisons": results}


async def compare_prices(query: str, pincode: str = "560087") -> dict:
    """Compare prices for a product across all platforms."""
    
    combined_results = []
    platforms_with_results = 0
    
    # Initialize all scrapers
    amazon = AmazonScraper(pincode)
    amazon_fresh = AmazonFreshScraper(pincode)
    flipkart = FlipkartScraper(pincode)
    flipkart_minutes = FlipkartMinutesScraper(pincode)
    zepto = ZeptoScraper(pincode)
    bigbasket = BigBasketScraper(pincode)
    jiomart_quick = JioMartQuickScraper(pincode)
    jiomart = JioMartScraper(pincode)
    
    # Helper to add results
    def add_results(name: str, results: list):
        nonlocal platforms_with_results
        if results:
            platforms_with_results += 1
            for result in results:
                combined_results.append(asdict(result))
    
    # Run all HTTP-based scrapers concurrently
    async def run_scraper(name, scraper):
        try:
            results = await asyncio.wait_for(scraper.search(query), timeout=25.0)
            return name, results
        except asyncio.TimeoutError:
            print(f"{name}: TIMEOUT")
            return name, []
        except Exception as e:
            print(f"{name}: ERROR - {e}")
            return name, []
    
    # Create tasks for all scrapers
    tasks = [
        run_scraper("Amazon Fresh", amazon_fresh),
        run_scraper("Flipkart Minutes", flipkart_minutes),
        run_scraper("JioMart Quick", jiomart_quick),
        run_scraper("BigBasket", bigbasket),
        run_scraper("Amazon", amazon),
        run_scraper("Flipkart", flipkart),
        run_scraper("JioMart", jiomart),
    ]
    
    # Run HTTP scrapers concurrently
    http_results = await asyncio.gather(*tasks)
    
    for name, results in http_results:
        add_results(name, results)
    
    # Run Zepto separately (uses Playwright browser)
    try:
        zepto_results = await asyncio.wait_for(zepto.search(query), timeout=40.0)
        add_results("Zepto", zepto_results)
    except asyncio.TimeoutError:
        print("Zepto: TIMEOUT")
    except Exception as e:
        print(f"Zepto: ERROR - {e}")
    
    # Sort results by platform in desired order
    platform_order = {
        "Amazon Fresh": 1,
        "Flipkart Minutes": 2,
        "JioMart Quick": 3,
        "BigBasket": 4,
        "Zepto": 5,
        "Amazon": 6,
        "Flipkart": 7,
        "JioMart": 8,
        "Blinkit": 9,
        "Instamart": 10,
    }
    combined_results.sort(key=lambda x: (platform_order.get(x.get("platform", ""), 99), x.get("price", 0)))
    
    # Find lowest price
    lowest_price = None
    if combined_results:
        available_results = [r for r in combined_results if r.get("available", True) and r.get("price", 0) > 0]
        if available_results:
            lowest_price = min(available_results, key=lambda x: x["price"])
    
    return {
        "query": query,
        "results": combined_results,
        "lowest_price": lowest_price,
        "total_platforms": platforms_with_results
    }


@app.get("/api/platforms")
async def get_platforms():
    """Get list of supported platforms."""
    return {
        "platforms": [
            {"name": "Amazon Fresh", "type": "quick-commerce", "delivery": "2-4 hours", "color": "#5EA03E"},
            {"name": "Flipkart Minutes", "type": "quick-commerce", "delivery": "10-45 mins", "color": "#FFCE00"},
            {"name": "JioMart Quick", "type": "quick-commerce", "delivery": "10-30 mins", "color": "#0078AD"},
            {"name": "BigBasket", "type": "quick-commerce", "delivery": "2-4 hours", "color": "#84C225"},
            {"name": "Zepto", "type": "quick-commerce", "delivery": "10-15 mins", "color": "#8B5CF6"},
            {"name": "Amazon", "type": "e-commerce", "delivery": "1-3 days", "color": "#FF9900"},
            {"name": "Flipkart", "type": "e-commerce", "delivery": "2-4 days", "color": "#2874F0"},
            {"name": "JioMart", "type": "e-commerce", "delivery": "1-3 days", "color": "#0078AD"},
            {"name": "Instamart", "type": "quick-commerce", "delivery": "15-30 mins", "color": "#FC8019"},
            {"name": "Blinkit", "type": "quick-commerce", "delivery": "8-12 mins", "color": "#F8CB46"},
        ]
    }


@app.get("/api/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    return cache.get_stats()


@app.post("/api/cache/clear")
async def cache_clear():
    """Clear all cache entries."""
    cache.clear()
    return {"status": "cleared", "message": "Cache cleared successfully"}


# ========== BACKEND PLAYWRIGHT SCRAPE (For when device extraction fails) ==========

class PlatformScrapeRequest(BaseModel):
    """Request to scrape a single platform using Playwright."""
    platform: str
    query: str
    pincode: str = "560001"


@app.post("/api/scrape/platform")
async def scrape_platform(request: PlatformScrapeRequest):
    """
    Scrape a single platform using Playwright browser automation.
    
    This endpoint is called by the mobile app when device-side extraction fails.
    It uses a full headless browser which is more reliable than WebView.
    
    Platforms: Zepto, Blinkit, BigBasket, Instamart, Flipkart, Flipkart Minutes,
               Amazon, Amazon Fresh, JioMart, JioMart Quick
    """
    platform_map = {
        "zepto": ZeptoScraper,
        "blinkit": BlinkitScraper,
        "bigbasket": BigBasketScraper,
        "instamart": InstamartScraper,
        "flipkart": FlipkartScraper,
        "flipkart minutes": FlipkartMinutesScraper,
        "amazon": AmazonScraper,
        "amazon fresh": AmazonFreshScraper,
        "jiomart": JioMartScraper,
        "jiomart quick": JioMartQuickScraper,
    }
    
    platform_key = request.platform.lower().strip()
    scraper_class = platform_map.get(platform_key)
    
    if not scraper_class:
        return {
            "success": False,
            "platform": request.platform,
            "error": f"Unknown platform: {request.platform}",
            "products": []
        }
    
    try:
        scraper = scraper_class(request.pincode)
        results = await asyncio.wait_for(scraper.search(request.query), timeout=30.0)
        products = [asdict(r) for r in results] if results else []
        
        # Cache successful results
        if products:
            cache.set(request.platform, request.query, request.pincode, products)
        
        return {
            "success": True,
            "platform": request.platform,
            "query": request.query,
            "products": products,
            "count": len(products)
        }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "platform": request.platform,
            "error": "Timeout - platform took too long to respond",
            "products": []
        }
    except Exception as e:
        return {
            "success": False,
            "platform": request.platform,
            "error": str(e),
            "products": []
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "price-comparator"}


# ============================================================================
# Analytics Dashboard Endpoints
# ============================================================================

from app.analytics import (
    ScrapeLogRequest, BulkLogRequest, DashboardQueryRequest,
    log_scrape_event, log_bulk_events, get_dashboard_data,
    get_recent_logs, get_all_devices
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
    start_date: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD)")
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
    limit: int = Query(100, description="Max logs to return"),
    platform: str = Query(None, description="Filter by platform")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

