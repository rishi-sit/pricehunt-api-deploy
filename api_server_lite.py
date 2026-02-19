"""
PriceHunt API Lite - AI-powered filtering AND fallback scraping
Designed for free tier hosting (Railway, Render, etc.)

This version:
1. Receives scraped products from Android app
2. Uses Gemini AI for smart filtering and product matching
3. Provides AI-powered HTML extraction as FALLBACK when client scraping fails
"""
import asyncio
from dataclasses import asdict
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# AI-powered modules
from app.smart_search import get_smart_search
from app.product_matcher import get_product_matcher
from app.ai_service import get_ai_service as get_gemini_service
from app.ai_scraper import get_ai_scraper
from app.analytics import (
    AIProcessingEventRequest, log_ai_processing_event
)
from app.scrapers import (
    AmazonScraper,
    AmazonFreshScraper,
    BlinkitScraper,
    BigBasketScraper,
    FlipkartScraper,
    FlipkartMinutesScraper,
    InstamartScraper,
    JioMartQuickScraper,
    JioMartScraper,
    ZeptoScraper,
)

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
    session_id: Optional[str] = None  # For analytics tracking
    device_id: Optional[str] = None  # For analytics tracking


class MatchProductsRequest(BaseModel):
    products: List[ProductInput]
    session_id: Optional[str] = None  # For analytics tracking
    device_id: Optional[str] = None  # For analytics tracking


class AIExtractRequest(BaseModel):
    """Request for AI-powered HTML extraction (fallback scraping)"""
    html: str
    platform: str
    search_query: str
    base_url: str
    device_id: Optional[str] = None  # For analytics tracking
    session_id: Optional[str] = None  # For session-based analytics


class MultiPlatformExtractRequest(BaseModel):
    """Request for extracting from multiple platform HTMLs"""
    platforms: List[Dict[str, str]]  # [{"platform": str, "html": str, "base_url": str}]
    search_query: str
    device_id: Optional[str] = None  # For analytics tracking
    session_id: Optional[str] = None  # For session-based analytics


class PlatformScrapeRequest(BaseModel):
    """Request to scrape a single platform using backend Playwright."""
    platform: str
    query: str
    pincode: str = "560001"


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """API info"""
    return {
        "name": "PriceHunt API Lite",
        "version": "2.2.0-lite",
        "description": "AI-powered filtering + fallback scraping + intelligent suggestions",
        "ai_enabled": gemini.is_available(),
        "ai_scraper_enabled": ai_scraper.is_available(),
        "endpoints": {
            "smart_search": "POST /api/smart-search",
            "match_products": "POST /api/match-products",
            "combined": "POST /api/smart-search-and-match",
            "understand_query": "GET /api/understand-query",
            "suggestions": "GET /api/suggestions (NEW - intelligent search suggestions)",
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


@app.get("/api/quota-stats")
async def get_quota_stats():
    """
    Get current AI quota statistics.
    Shows which providers are available, exhausted, and request counts.
    """
    return {
        "ai_service": gemini.get_quota_stats(),
        "ai_scraper": ai_scraper.get_quota_stats() if hasattr(ai_scraper, 'get_quota_stats') else {},
        "ai_available": gemini.is_available(),
        "ai_scraper_available": ai_scraper.is_available()
    }


@app.get("/api/compare-models")
async def compare_ai_models(
    models: str = "gemini-2.5-flash,gemini-2.5-flash-lite",
    test_query: str = "strawberry"
):
    """
    Compare multiple AI models on the same test cases.
    
    Helps identify which model gives the best results for:
    1. Overall accuracy (correct relevant/irrelevant classification)
    2. Best deal accuracy (returns correct best deal)
    3. Latency (response time)
    
    Args:
        models: Comma-separated list of model names to test
        test_query: Query to test (default: strawberry)
    
    Returns:
        Comparison metrics for each model
    """
    import statistics
    
    # Test products for the query
    test_products_map = {
        "strawberry": [
            {"name": "Fresh Strawberry 200g", "price": 99, "platform": "Zepto"},
            {"name": "Strawberry Shake 200ml", "price": 45, "platform": "BigBasket"},
            {"name": "Strawberry Jam 200g", "price": 89, "platform": "Amazon"},
            {"name": "American Strawberry 200g Pack", "price": 149, "platform": "Blinkit"},
            {"name": "Fresh Strawberries Premium 250g", "price": 129, "platform": "Instamart"},
            {"name": "Strawberry Flavoured Milk 200ml", "price": 35, "platform": "JioMart"},
        ],
        "banana": [
            {"name": "Banana Robusta 1kg", "price": 49, "platform": "Zepto"},
            {"name": "Yellaki Banana 12 pcs", "price": 59, "platform": "BigBasket"},
            {"name": "Banana Chips 150g", "price": 45, "platform": "Amazon"},
            {"name": "Cavendish Banana 6 pcs", "price": 45, "platform": "Instamart"},
            {"name": "Raw Banana 500g", "price": 35, "platform": "JioMart"},
            {"name": "Banana Wafer 200g", "price": 65, "platform": "Blinkit"},
        ],
        "milk": [
            {"name": "Amul Taaza Toned Milk 500ml", "price": 28, "platform": "Zepto"},
            {"name": "Mother Dairy Full Cream Milk 1L", "price": 68, "platform": "BigBasket"},
            {"name": "Cadbury Dairy Milk Chocolate 50g", "price": 50, "platform": "Blinkit"},
            {"name": "Nestle Milkshake Strawberry 180ml", "price": 35, "platform": "JioMart"},
            {"name": "Nandini Milk 500ml", "price": 25, "platform": "Instamart"},
        ],
        "apple": [
            {"name": "Fresh Apple Red Delicious 1kg", "price": 180, "platform": "BigBasket"},
            {"name": "Apple iPhone 15 Case", "price": 999, "platform": "Amazon"},
            {"name": "Real Apple Juice 1L", "price": 99, "platform": "Zepto"},
            {"name": "Organic Green Apple 500g", "price": 120, "platform": "JioMart"},
            {"name": "Pineapple Fresh 1kg", "price": 90, "platform": "Instamart"},
        ],
    }
    
    expected_relevant_map = {
        "strawberry": ["Fresh Strawberry 200g", "American Strawberry 200g Pack", "Fresh Strawberries Premium 250g"],
        "banana": ["Banana Robusta 1kg", "Yellaki Banana 12 pcs", "Cavendish Banana 6 pcs", "Raw Banana 500g"],
        "milk": ["Amul Taaza Toned Milk 500ml", "Mother Dairy Full Cream Milk 1L", "Nandini Milk 500ml"],
        "apple": ["Fresh Apple Red Delicious 1kg", "Organic Green Apple 500g"],
    }
    
    test_products = test_products_map.get(test_query.lower(), test_products_map["strawberry"])
    expected_relevant = expected_relevant_map.get(test_query.lower(), expected_relevant_map["strawberry"])
    expected_set = {n.lower().strip() for n in expected_relevant}
    all_items = {p.get("name", "").lower().strip() for p in test_products}
    
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    results = []
    
    for model in model_list:
        # Determine provider from model name
        if "gemini" in model.lower() or "gemma" in model.lower():
            provider = "gemini"
        elif "llama" in model.lower() or "mixtral" in model.lower():
            provider = "groq"
        elif "mistral" in model.lower():
            provider = "mistral"
        else:
            provider = None
        
        start_time = time.monotonic()
        
        try:
            ai_result = await gemini.filter_relevant_products_with_provider(
                query=test_query,
                products=test_products,
                strict_mode=True,
                provider=provider,
                model=model
            )
            
            latency_ms = int((time.monotonic() - start_time) * 1000)
            
            # Calculate metrics
            relevant_products = ai_result.get("relevant_products", [])
            predicted_set = {p.get("name", "").lower().strip() for p in relevant_products}
            
            tp = len(predicted_set & expected_set)
            fp = len(predicted_set - expected_set)
            fn = len(expected_set - predicted_set)
            tn = len((all_items - predicted_set) & (all_items - expected_set))
            
            total = len(all_items)
            accuracy = (tp + tn) / total if total else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            
            # Check best deal
            best_deal = ai_result.get("best_deal")
            best_deal_correct = False
            best_deal_name = None
            if best_deal:
                best_deal_name = best_deal.get("name")
                bd_lower = (best_deal_name or "").lower()
                # Best deal should be a fresh product (in expected_set)
                best_deal_correct = any(exp in bd_lower or bd_lower in exp 
                                       for exp in expected_set)
            
            ai_meta = ai_result.get("ai_meta", {})
            
            results.append({
                "model": model,
                "provider": ai_meta.get("provider", provider),
                "model_used": ai_meta.get("model", model),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "latency_ms": latency_ms,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "best_deal_correct": best_deal_correct,
                "best_deal_name": best_deal_name,
                "predicted_relevant": sorted(predicted_set),
                "ai_powered": ai_result.get("ai_powered", False),
                "error": None
            })
        except Exception as e:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            results.append({
                "model": model,
                "provider": provider,
                "error": str(e),
                "latency_ms": latency_ms,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "best_deal_correct": False
            })
    
    # Rank models
    valid_results = [r for r in results if not r.get("error")]
    if valid_results:
        best_accuracy = max(valid_results, key=lambda r: r.get("accuracy", 0))
        best_f1 = max(valid_results, key=lambda r: r.get("f1_score", 0))
        fastest = min(valid_results, key=lambda r: r.get("latency_ms", float("inf")))
        best_deal_models = [r for r in valid_results if r.get("best_deal_correct")]
    else:
        best_accuracy = best_f1 = fastest = None
        best_deal_models = []
    
    return {
        "test_query": test_query,
        "models_tested": model_list,
        "expected_relevant": expected_relevant,
        "results": results,
        "rankings": {
            "best_accuracy": best_accuracy["model"] if best_accuracy else None,
            "best_f1": best_f1["model"] if best_f1 else None,
            "fastest": fastest["model"] if fastest else None,
            "best_deal_correct_models": [r["model"] for r in best_deal_models]
        },
        "recommendations": {
            "for_accuracy": best_accuracy["model"] if best_accuracy else "No model available",
            "for_speed": fastest["model"] if fastest else "No model available",
            "for_best_deal": best_deal_models[0]["model"] if best_deal_models else "No model returned correct best deal"
        }
    }


@app.get("/api/scraping-metrics")
async def get_scraping_metrics(days: int = 7, device_id: Optional[str] = None):
    """
    Get scraping metrics showing device vs AI fallback success rates.
    
    This helps analyze:
    1. Which platforms need device-side extractor updates
    2. AI fallback usage patterns
    3. Overall scraping success rates
    
    Args:
        days: Number of days to look back (default: 7)
        device_id: Optional filter by device ID
    
    Returns:
        Aggregated metrics for scraping performance
    """
    from app.analytics import get_db
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get platform scrape events
        query = """
            SELECT 
                platform,
                scrape_source,
                COUNT(*) as count,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                AVG(products_found) as avg_products,
                AVG(html_size_kb) as avg_html_kb,
                AVG(latency_ms) as avg_latency_ms
            FROM platform_scrape_events
            WHERE created_at >= ? AND created_at <= ?
        """
        params = [start_date.isoformat(), end_date.isoformat()]
        
        if device_id:
            query += " AND device_id = ?"
            params.append(device_id)
        
        query += " GROUP BY platform, scrape_source ORDER BY platform, scrape_source"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        platform_metrics = {}
        total_device = 0
        total_ai = 0
        total_playwright = 0
        total_cache = 0
        
        for row in rows:
            platform = row["platform"]
            source = row["scrape_source"]
            count = row["count"]
            
            if platform not in platform_metrics:
                platform_metrics[platform] = {
                    "total": 0,
                    "device": 0,
                    "ai_fallback": 0,
                    "playwright": 0,
                    "cache": 0,
                    "device_success_rate": 0.0,
                    "avg_products": 0.0
                }
            
            platform_metrics[platform]["total"] += count
            platform_metrics[platform][source] = count
            
            if source == "device":
                total_device += count
                platform_metrics[platform]["device_success_rate"] = (
                    row["success_count"] / count if count > 0 else 0.0
                )
            elif source == "ai_fallback":
                total_ai += count
            elif source == "playwright":
                total_playwright += count
            elif source == "cache":
                total_cache += count
        
        # Get AI processing metrics
        ai_query = """
            SELECT 
                ai_provider,
                ai_model,
                COUNT(*) as count,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                AVG(latency_ms) as avg_latency_ms,
                AVG(products_output) as avg_products_output
            FROM ai_processing_events
            WHERE created_at >= ? AND created_at <= ?
        """
        ai_params = [start_date.isoformat(), end_date.isoformat()]
        if device_id:
            ai_query += " AND device_id = ?"
            ai_params.append(device_id)
        ai_query += " GROUP BY ai_provider, ai_model ORDER BY count DESC"
        
        cursor.execute(ai_query, ai_params)
        ai_rows = cursor.fetchall()
        
        ai_metrics = []
        for row in ai_rows:
            ai_metrics.append({
                "provider": row["ai_provider"],
                "model": row["ai_model"],
                "requests": row["count"],
                "success_rate": row["success_count"] / row["count"] if row["count"] > 0 else 0.0,
                "avg_latency_ms": int(row["avg_latency_ms"] or 0),
                "avg_products_output": float(row["avg_products_output"] or 0)
            })
        
        total_scrapes = total_device + total_ai + total_playwright + total_cache
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            },
            "overall": {
                "total_scrapes": total_scrapes,
                "device_rate": total_device / total_scrapes if total_scrapes > 0 else 0.0,
                "ai_fallback_rate": total_ai / total_scrapes if total_scrapes > 0 else 0.0,
                "playwright_rate": total_playwright / total_scrapes if total_scrapes > 0 else 0.0,
                "cache_rate": total_cache / total_scrapes if total_scrapes > 0 else 0.0
            },
            "by_platform": platform_metrics,
            "ai_models": ai_metrics,
            "recommendations": _generate_scraping_recommendations(platform_metrics, ai_metrics)
        }


def _generate_scraping_recommendations(platform_metrics: Dict, ai_metrics: List) -> List[str]:
    """Generate recommendations based on scraping metrics."""
    recommendations = []
    
    # Check for platforms with high AI fallback rate
    high_ai_platforms = []
    for platform, metrics in platform_metrics.items():
        total = metrics.get("total", 0)
        ai = metrics.get("ai_fallback", 0)
        if total > 0 and (ai / total) > 0.5:
            high_ai_platforms.append(platform)
    
    if high_ai_platforms:
        recommendations.append(
            f"Platforms with high AI fallback (>50%): {', '.join(high_ai_platforms)}. "
            "Consider updating device-side extractors for these platforms."
        )
    
    # Check for low device success rates
    low_success_platforms = []
    for platform, metrics in platform_metrics.items():
        if metrics.get("device_success_rate", 0) < 0.5 and metrics.get("device", 0) > 5:
            low_success_platforms.append(platform)
    
    if low_success_platforms:
        recommendations.append(
            f"Platforms with low device success rate (<50%): {', '.join(low_success_platforms)}. "
            "HTML structure may have changed."
        )
    
    # AI model recommendations
    if ai_metrics:
        best_model = max(ai_metrics, key=lambda m: m.get("success_rate", 0))
        if best_model.get("success_rate", 0) > 0.8:
            recommendations.append(
                f"Best performing AI model: {best_model['provider']}/{best_model['model']} "
                f"(success rate: {best_model['success_rate']:.0%})"
            )
    
    if not recommendations:
        recommendations.append("Scraping is performing well! No immediate action needed.")
    
    return recommendations


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

        # Log AI filtering event (Stage 3: AI Filtering)
        if request.session_id or request.device_id:
            try:
                ai_meta = result.ai_meta or {}
                log_ai_processing_event(AIProcessingEventRequest(
                    session_id=request.session_id,
                    device_id=request.device_id,
                    endpoint="smart-search",
                    platform=None,  # Applies to all platforms
                    ai_provider=ai_meta.get("provider", "unknown"),
                    ai_model=ai_meta.get("model", "unknown"),
                    is_fallback=ai_meta.get("fallback_reason") is not None,
                    fallback_reason=ai_meta.get("fallback_reason"),
                    products_input=len(products_dict),
                    products_output=result.total_found,
                    latency_ms=filter_ms,
                    success=result.ai_powered,
                    metadata={"query_understanding": result.query_understanding, "platform_counts": platform_counts}
                ))
            except Exception as log_err:
                print(f"⚠️ Failed to log smart-search AI event: {log_err}")

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

        # Log AI matching event (Stage 4: Product Matching)
        if request.session_id or request.device_id:
            try:
                ai_meta = result.ai_meta or {}
                log_ai_processing_event(AIProcessingEventRequest(
                    session_id=request.session_id,
                    device_id=request.device_id,
                    endpoint="match-products",
                    platform=None,  # Applies to all platforms
                    ai_provider=ai_meta.get("provider", "unknown"),
                    ai_model=ai_meta.get("model", "unknown"),
                    is_fallback=ai_meta.get("fallback_reason") is not None,
                    fallback_reason=ai_meta.get("fallback_reason"),
                    products_input=len(products_dict),
                    products_output=result.total_matched,
                    latency_ms=match_ms,
                    success=result.ai_powered,
                    metadata={"total_groups": result.total_groups}
                ))
            except Exception as log_err:
                print(f"⚠️ Failed to log match-products AI event: {log_err}")

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

        # Log AI filtering event (Stage 3: AI Filtering)
        if request.session_id or request.device_id:
            try:
                filter_ai_meta = filter_result.ai_meta or {}
                log_ai_processing_event(AIProcessingEventRequest(
                    session_id=request.session_id,
                    device_id=request.device_id,
                    endpoint="smart-search-and-match-filter",
                    platform=None,
                    ai_provider=filter_ai_meta.get("provider", "unknown"),
                    ai_model=filter_ai_meta.get("model", "unknown"),
                    is_fallback=filter_ai_meta.get("fallback_reason") is not None,
                    fallback_reason=filter_ai_meta.get("fallback_reason"),
                    products_input=len(products_dict),
                    products_output=filter_result.total_found,
                    latency_ms=filter_ms,
                    success=filter_result.ai_powered,
                    metadata={"query_understanding": filter_result.query_understanding}
                ))
            except Exception as log_err:
                print(f"⚠️ Failed to log smart-search-and-match filter AI event: {log_err}")

        # Log AI matching event (Stage 4: Product Matching)
        if request.session_id or request.device_id:
            try:
                match_ai_meta = match_result.ai_meta or {}
                log_ai_processing_event(AIProcessingEventRequest(
                    session_id=request.session_id,
                    device_id=request.device_id,
                    endpoint="smart-search-and-match-match",
                    platform=None,
                    ai_provider=match_ai_meta.get("provider", "unknown"),
                    ai_model=match_ai_meta.get("model", "unknown"),
                    is_fallback=match_ai_meta.get("fallback_reason") is not None,
                    fallback_reason=match_ai_meta.get("fallback_reason"),
                    products_input=len(filtered_products),
                    products_output=match_result.total_matched,
                    latency_ms=match_ms,
                    success=match_result.ai_powered,
                    metadata={"total_groups": match_result.total_groups}
                ))
            except Exception as log_err:
                print(f"⚠️ Failed to log smart-search-and-match AI event: {log_err}")

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
        
        input_size_kb = len(request.html) / 1024
        
        # Step 1: Extract products from HTML
        result = await ai_scraper.extract_products_from_html(
            html=request.html,
            platform=request.platform,
            search_query=request.search_query,
            base_url=request.base_url,
            device_id=request.device_id
        )
        
        extracted_products = result.get("products", [])
        
        # Log AI extraction event (Stage 2: AI Extraction)
        if request.session_id or request.device_id:
            try:
                extraction_ai_meta = {
                    "provider": result.get("provider", "unknown"),
                    "model": result.get("model", "unknown"),
                }
                log_ai_processing_event(AIProcessingEventRequest(
                    session_id=request.session_id,
                    device_id=request.device_id,
                    endpoint="ai-extract",
                    platform=request.platform,
                    ai_provider=extraction_ai_meta.get("provider", "gemini"),
                    ai_model=extraction_ai_meta.get("model", "unknown"),
                    is_fallback=False,
                    input_size_kb=input_size_kb,
                    products_input=0,  # No input products for extraction
                    products_output=len(extracted_products),
                    latency_ms=result.get("latency_ms", 0),
                    success=len(extracted_products) > 0,
                    error_message=result.get("error"),
                    metadata={"extraction_method": result.get("extraction_method"), "confidence": result.get("confidence")}
                ))
            except Exception as log_err:
                print(f"⚠️ Failed to log AI extraction event: {log_err}")
        
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
        filter_ai_meta = filter_result.get("ai_meta", {})
        
        # Log AI filtering event (Stage 3: AI Filtering)
        if request.session_id or request.device_id:
            try:
                log_ai_processing_event(AIProcessingEventRequest(
                    session_id=request.session_id,
                    device_id=request.device_id,
                    endpoint="ai-filter",
                    platform=request.platform,
                    ai_provider=filter_ai_meta.get("provider", "unknown"),
                    ai_model=filter_ai_meta.get("model", "unknown"),
                    is_fallback=filter_ai_meta.get("fallback_reason") is not None,
                    fallback_reason=filter_ai_meta.get("fallback_reason"),
                    products_input=len(extracted_products),
                    products_output=len(relevant_products),
                    latency_ms=filter_ai_meta.get("latency_ms", 0),
                    success=filter_result.get("ai_powered", False),
                    metadata={"query_understanding": filter_result.get("query_understanding")}
                ))
            except Exception as log_err:
                print(f"⚠️ Failed to log AI filter event: {log_err}")
        
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
            "ai_meta": filter_ai_meta,
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


@app.post("/api/scrape/platform")
async def scrape_platform(request: PlatformScrapeRequest):
    """Scrape a single platform on the backend when device-side scrape fails."""
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
            "products": [],
        }

    try:
        scraper = scraper_class(request.pincode)
        results = await asyncio.wait_for(scraper.search(request.query), timeout=30.0)
        products = [asdict(r) for r in results] if results else []

        return {
            "success": True,
            "platform": request.platform,
            "query": request.query,
            "products": products,
            "count": len(products),
        }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "platform": request.platform,
            "error": "Timeout - platform took too long to respond",
            "products": [],
        }
    except Exception as e:
        return {
            "success": False,
            "platform": request.platform,
            "error": f"Error: {str(e)}",
            "products": [],
        }


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
        
        # Log AI extraction events for each platform (Stage 2: AI Extraction)
        if request.session_id or request.device_id:
            for platform, result in extraction_result.get("results", {}).items():
                try:
                    log_ai_processing_event(AIProcessingEventRequest(
                        session_id=request.session_id,
                        device_id=request.device_id,
                        endpoint="smart-extract-extraction",
                        platform=platform,
                        ai_provider=result.get("provider", "gemini"),
                        ai_model=result.get("model", "unknown"),
                        is_fallback=False,
                        products_input=0,
                        products_output=len(result.get("products", [])),
                        latency_ms=result.get("latency_ms", 0),
                        success=len(result.get("products", [])) > 0,
                        error_message=result.get("error"),
                        metadata={"extraction_method": result.get("extraction_method"), "confidence": result.get("confidence")}
                    ))
                except Exception as log_err:
                    print(f"⚠️ Failed to log smart-extract extraction AI event for {platform}: {log_err}")
        
        # Step 2: Filter relevant products
        filter_result = await smart_search.search(
            query=request.search_query,
            products=all_products
        )
        
        filtered_products = filter_result.get("products", all_products)
        
        # Log AI filtering event (Stage 3: AI Filtering)
        if request.session_id or request.device_id:
            try:
                filter_ai_meta = filter_result.get("ai_meta") or {}
                log_ai_processing_event(AIProcessingEventRequest(
                    session_id=request.session_id,
                    device_id=request.device_id,
                    endpoint="smart-extract-filter",
                    platform=None,
                    ai_provider=filter_ai_meta.get("provider", "unknown"),
                    ai_model=filter_ai_meta.get("model", "unknown"),
                    is_fallback=filter_ai_meta.get("fallback_reason") is not None,
                    fallback_reason=filter_ai_meta.get("fallback_reason"),
                    products_input=len(all_products),
                    products_output=len(filtered_products),
                    latency_ms=filter_ai_meta.get("latency_ms", 0),
                    success=filter_result.get("ai_powered", False),
                    metadata={"query_understanding": filter_result.get("query_understanding")}
                ))
            except Exception as log_err:
                print(f"⚠️ Failed to log smart-extract filter AI event: {log_err}")
        
        # Step 3: Match similar products
        match_result = await product_matcher.match_products(filtered_products)
        
        # Log AI matching event (Stage 4: Product Matching)
        if request.session_id or request.device_id:
            try:
                match_ai_meta = match_result.ai_meta if hasattr(match_result, 'ai_meta') else {}
                log_ai_processing_event(AIProcessingEventRequest(
                    session_id=request.session_id,
                    device_id=request.device_id,
                    endpoint="smart-extract-match",
                    platform=None,
                    ai_provider=match_ai_meta.get("provider", "unknown") if match_ai_meta else "unknown",
                    ai_model=match_ai_meta.get("model", "unknown") if match_ai_meta else "unknown",
                    is_fallback=False,
                    products_input=len(filtered_products),
                    products_output=match_result.total_matched if hasattr(match_result, 'total_matched') else len(match_result.get("groups", [])),
                    latency_ms=match_ai_meta.get("latency_ms", 0) if match_ai_meta else 0,
                    success=match_result.ai_powered if hasattr(match_result, 'ai_powered') else True,
                    metadata={"total_groups": match_result.total_groups if hasattr(match_result, 'total_groups') else len(match_result.get("groups", []))}
                ))
            except Exception as log_err:
                print(f"⚠️ Failed to log smart-extract match AI event: {log_err}")
        
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
    get_ai_processing_stats, get_combined_dashboard, get_ai_quota_stats,
    # NEW: App-wide holistic stats
    get_app_wide_stats, get_recent_sessions,
    # NEW: Session-based analytics
    CreateSessionRequest, UpdateSessionRequest, PlatformScrapeEventRequest,
    AIProcessingEventRequest, create_session, update_session,
    log_platform_scrape_event, log_ai_processing_event, get_session_detail,
    get_device_sessions, get_session_pipeline_visualization,
    # NEW: AI accuracy tracking
    AIAccuracyLogRequest, log_ai_accuracy, get_ai_model_accuracy_stats
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


@app.get("/api/analytics/ai-quota")
async def get_ai_quota():
    """
    Get AI quota usage stats across ALL devices.
    
    Shows total hits per AI provider to track quota limits:
    - Groq: 6000 requests/day free tier
    - Mistral: ~10K requests/day (1B tokens/month)
    - Gemini: 1500/day, 60/minute free tier
    
    Returns:
    - Total requests all-time per provider
    - Today's requests per provider
    - Last hour requests (for rate limiting)
    - Quota remaining estimates
    """
    try:
        quota_stats = get_ai_quota_stats()
        return {
            "success": True,
            "data": quota_stats
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.get("/api/analytics/debug-db")
async def debug_db():
    """Debug endpoint to check DB row format."""
    from app.analytics import get_db, get_cursor, fetchone_as_dict
    try:
        with get_db() as conn:
            cursor = get_cursor(conn)
            cursor.execute("SELECT 1 as test_val, 'hello' as test_str")
            desc_before = cursor.description
            row = cursor.fetchone()
            desc_after = cursor.description
            
            row_info = {
                "row_type": str(type(row)),
                "row_repr": repr(row)[:500],
                "desc_before": [str(d) for d in desc_before] if desc_before else None,
                "desc_after": [str(d) for d in desc_after] if desc_after else None,
                "row_dir": [a for a in dir(row) if not a.startswith('_')][:20] if row else [],
            }
            
            # Try index access
            try:
                row_info["index_0"] = str(row[0])
                row_info["index_1"] = str(row[1])
            except Exception as e:
                row_info["index_error"] = str(e)
            
            # Test helper
            cursor.execute("SELECT 'world' as greet")
            try:
                helper_result = fetchone_as_dict(cursor)
                row_info["helper_result"] = helper_result
            except Exception as e:
                row_info["helper_error"] = str(e)
            
            return {"success": True, "deployed": "v8-sql-parsing-fix", **row_info}
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "tb": traceback.format_exc()}


@app.get("/api/analytics/app-wide")
async def get_app_wide_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get HOLISTIC app-wide statistics across ALL devices.
    
    This is the "bird's eye view" dashboard showing:
    - Total searches, products, AI calls across all devices
    - Platform-wise failure rates (identify problematic platforms)
    - Best deal success rates
    - AI relevance extraction accuracy
    - Scrape source distribution (device vs AI fallback vs playwright)
    
    Use this for:
    - Monitoring overall system health
    - Identifying quota issues before they hit limits
    - Spotting platform-specific scraping failures
    - Measuring AI extraction accuracy
    """
    try:
        stats = get_app_wide_stats(start_date, end_date)
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/analytics/recent-sessions")
async def get_recent_sessions_endpoint(
    device_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50
):
    """
    Get recent search sessions for dashboard display.
    
    If device_id is omitted, returns sessions across ALL devices.
    Each session includes summary info and a link to the pipeline drill-down view.
    
    Use this for:
    - Viewing recent searches across the app
    - Quick access to session pipeline debugging
    - Identifying failed sessions
    """
    try:
        sessions = get_recent_sessions(device_id, limit, start_date, end_date)
        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/analytics/ai-accuracy")
async def get_ai_accuracy_stats_endpoint(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get AI model accuracy comparison stats.
    
    Compares different AI models (Groq, Gemini, Mistral, Cerebras, Together) on:
    - Accuracy Score: % of kept products that are high relevance (score >= 80)
    - Best Deal Quality: % of best deals that are exact/close matches
    - Average latency
    - Total calls
    
    Use this to determine:
    - Which AI model filters most accurately
    - Which model provides best deal accuracy
    - Performance/accuracy trade-offs
    """
    try:
        stats = get_ai_model_accuracy_stats(start_date, end_date)
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/analytics/ai-accuracy/log")
async def log_ai_accuracy_endpoint(log: AIAccuracyLogRequest):
    """
    Log AI accuracy metrics for model comparison.
    
    Called after AI filtering completes with metrics:
    - search_query: What user searched for
    - ai_provider/model: Which AI was used
    - products_kept/filtered: Filtering stats
    - high/medium/low_relevance_count: Score distribution
    - best_deal_relevance_score: Score of the chosen best deal
    - best_deal_reason: Why it was chosen (exact_match, close_match, lowest_price)
    """
    try:
        log_id = log_ai_accuracy(log)
        return {
            "success": True,
            "log_id": log_id,
            "message": "AI accuracy logged successfully"
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


# ============================================================================
# Session-Based Pipeline Analytics (NEW)
# ============================================================================

@app.post("/api/analytics/session/create")
async def create_search_session(request: CreateSessionRequest):
    """
    Create a new search session.
    
    Call this when a search starts on Android. Generate a UUID for session_id.
    All subsequent events (platform scrapes, AI calls) should include this session_id.
    """
    try:
        session_id = create_session(request)
        return {
            "success": True,
            "session_id": session_id,
            "message": "Session created"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/analytics/session/update")
async def update_search_session(request: UpdateSessionRequest):
    """
    Update session with final results after search completes.
    
    Include summary stats: total products, best deal, latency, etc.
    """
    try:
        updated = update_session(request)
        return {
            "success": True,
            "updated": updated,
            "message": "Session updated"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/analytics/session/platform-event")
async def log_platform_event(request: PlatformScrapeEventRequest):
    """
    Log a platform scrape event within a session.
    
    Call this after scraping each platform (Zepto, Blinkit, etc.)
    """
    try:
        event_id = log_platform_scrape_event(request)
        return {
            "success": True,
            "event_id": event_id,
            "message": "Platform event logged"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/analytics/session/ai-event")
async def log_ai_event(request: AIProcessingEventRequest):
    """
    Log an AI processing event within a session.
    
    Call this after each AI call (extraction, filtering, matching).
    """
    try:
        event_id = log_ai_processing_event(request)
        return {
            "success": True,
            "event_id": event_id,
            "message": "AI event logged"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/analytics/session/{session_id}")
async def get_session(session_id: str):
    """
    Get full session detail with all platform and AI events.
    
    Returns the complete journey of a single search.
    """
    try:
        detail = get_session_detail(session_id)
        if not detail:
            return {"success": False, "error": "Session not found"}
        return {"success": True, "data": detail}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/analytics/session/{session_id}/pipeline")
async def get_session_pipeline(session_id: str):
    """
    Get session data formatted for pipeline visualization.
    
    Returns stages: Scraping → AI Extraction → Filtering → Matching → Best Deal
    """
    try:
        pipeline = get_session_pipeline_visualization(session_id)
        if "error" in pipeline:
            return {"success": False, "error": pipeline["error"]}
        return {"success": True, "data": pipeline}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/analytics/sessions/{device_id}")
async def get_sessions_for_device(
    device_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50
):
    """
    Get recent sessions for a device.
    
    Returns list of sessions with summary info.
    """
    try:
        sessions = get_device_sessions(device_id, start_date, end_date, limit)
        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/analytics/ai-events/debug")
async def debug_ai_events(
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 20
):
    """
    Debug endpoint: Query AI processing events directly.
    """
    from app.analytics import get_db, get_cursor, fetchall_as_dicts
    
    try:
        with get_db() as conn:
            cursor = get_cursor(conn)
            
            if session_id:
                cursor.execute("""
                    SELECT * FROM ai_processing_events 
                    WHERE session_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (session_id, limit))
            elif device_id:
                cursor.execute("""
                    SELECT * FROM ai_processing_events 
                    WHERE device_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (device_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM ai_processing_events 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
            
            events = fetchall_as_dicts(cursor)
            
            return {
                "success": True,
                "count": len(events),
                "events": events
            }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ============================================================================
# Analytics Dashboard UI
# ============================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def analytics_dashboard():
    """
    Serve the Analytics Dashboard UI.
    
    Access at: https://pricehunt-hklm.onrender.com/dashboard
    
    Features:
    - Device selection dropdown
    - Date range filtering
    - Scrape source breakdown (Device vs AI Fallback vs Playwright)
    - Platform-wise statistics
    - AI provider usage stats (Groq vs Gemini vs Mistral)
    - Recent logs table
    """
    try:
        template_path = Path(__file__).parent / "app" / "templates" / "dashboard.html"
        if not template_path.exists():
            # Try alternate path
            template_path = Path(__file__).parent / "templates" / "dashboard.html"
        if not template_path.exists():
            return HTMLResponse(
                content="<h1>Dashboard template not found</h1><p>Please check deployment.</p>",
                status_code=404
            )
        with open(template_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Error loading dashboard</h1><p>{str(e)}</p>",
            status_code=500
        )


@app.get("/session", response_class=HTMLResponse)
async def session_pipeline_view():
    """
    Serve the Session Pipeline Visualization UI.
    
    Access at: https://pricehunt-hklm.onrender.com/session?id=<session_id>
    
    Shows the complete end-to-end journey of a single search:
    - Stage 1: Platform scraping (device/AI/Playwright per platform)
    - Stage 2: AI extraction calls
    - Stage 3: AI filtering
    - Stage 4: Product matching and best deal
    """
    try:
        template_path = Path(__file__).parent / "app" / "templates" / "session.html"
        if not template_path.exists():
            return HTMLResponse(
                content="<h1>Session template not found</h1>",
                status_code=404
            )
        with open(template_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Error loading session view</h1><p>{str(e)}</p>",
            status_code=500
        )


# ============== SUGGESTIONS API ==============

class SuggestionsResponse(BaseModel):
    """Response model for suggestions endpoint"""
    query: str
    category: Optional[str] = None
    suggestions: List[str]
    related: Optional[Dict[str, List[str]]] = None

# Initialize suggestions engine lazily
_suggestions_engine = None

def get_suggestions_engine():
    """Get or initialize suggestions engine"""
    global _suggestions_engine
    if _suggestions_engine is None:
        from app.suggestions import SuggestionsEngine
        _suggestions_engine = SuggestionsEngine()
    return _suggestions_engine

@app.get("/api/suggestions", response_model=SuggestionsResponse)
async def get_suggestions(
    query: str,
    max_suggestions: int = 5,
    pincode: str = "560001"
):
    """
    Get intelligent search suggestions based on user query
    
    Features:
    - Category-based suggestions (dairy, fruit, rice, oil, etc.)
    - Smart disambiguation (apple fruit vs Apple tech)
    - Search history integration
    - Graceful fallback to local generation
    
    Args:
        query: User's search query (e.g., "milk", "banana")
        max_suggestions: Maximum number of suggestions to return
        pincode: User's pincode for location-based suggestions
    
    Returns:
        SuggestionsResponse with intelligent suggestions and related products
    """
    try:
        engine = get_suggestions_engine()
        
        # Generate intelligent suggestions
        suggestions = engine.generate_suggestions(query, max_suggestions)
        category = engine.get_category(query)
        related = engine.get_related_products(query)
        
        return SuggestionsResponse(
            query=query,
            category=category,
            suggestions=suggestions,
            related=related
        )
    except Exception as e:
        # Fallback to basic suggestions if engine fails
        basic_suggestions = [
            f"{query} fresh",
            f"{query} organic", 
            f"{query} 1kg",
            f"{query} best quality",
            f"buy {query}"
        ][:max_suggestions]
        
        return SuggestionsResponse(
            query=query,
            category=None,
            suggestions=basic_suggestions,
            related=None
        )
