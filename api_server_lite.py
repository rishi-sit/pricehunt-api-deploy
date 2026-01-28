"""
PriceHunt API Lite - AI-powered filtering only (no scraping)
Designed for free tier hosting (Railway, Render, etc.)

This version receives scraped products from Android app and uses
Gemini AI for smart filtering and product matching.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import os

# AI-powered modules
from app.smart_search import get_smart_search
from app.product_matcher import get_product_matcher
from app.gemini_service import get_gemini_service

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


# ============== Pydantic Models ==============

class ProductInput(BaseModel):
    name: str
    price: float
    original_price: Optional[float] = None
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    platform: str
    quantity: Optional[str] = None
    unit: Optional[str] = None
    in_stock: bool = True


class SmartSearchRequest(BaseModel):
    query: str
    products: List[ProductInput]


class MatchProductsRequest(BaseModel):
    products: List[ProductInput]


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """API info"""
    return {
        "name": "PriceHunt API Lite",
        "version": "2.0.0-lite",
        "description": "AI-powered filtering only (Android does scraping)",
        "ai_enabled": gemini.is_available(),
        "endpoints": {
            "smart_search": "POST /api/smart-search",
            "match_products": "POST /api/match-products",
            "combined": "POST /api/smart-search-and-match",
            "understand_query": "GET /api/understand-query",
            "health": "GET /api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check with AI status"""
    return {
        "status": "healthy",
        "version": "2.0.0-lite",
        "ai_available": gemini.is_available(),
        "mode": "lite (no scraping)"
    }


@app.post("/api/smart-search")
async def smart_search_endpoint(request: SmartSearchRequest):
    """
    Filter products using AI to keep only relevant results.
    Example: "milk" query filters out "milkshake", "milkmade", etc.
    """
    try:
        products_dict = [p.model_dump() for p in request.products]
        
        result = await smart_search.search(
            query=request.query,
            products=products_dict
        )
        
        return {
            "query": request.query,
            "total_input": len(request.products),
            "total_filtered": len(result.get("products", [])),
            "ai_powered": result.get("ai_powered", False),
            "products": result.get("products", []),
            "best_deal": result.get("best_deal")
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
        products_dict = [p.model_dump() for p in request.products]
        
        result = await product_matcher.match_products(products_dict)
        
        return {
            "total_products": len(request.products),
            "groups_found": len(result.get("groups", [])),
            "ai_powered": result.get("ai_powered", False),
            "groups": result.get("groups", [])
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
        products_dict = [p.model_dump() for p in request.products]
        
        # Step 1: Smart filter
        filter_result = await smart_search.search(
            query=request.query,
            products=products_dict
        )
        
        filtered_products = filter_result.get("products", [])
        
        # Step 2: Match similar products
        match_result = await product_matcher.match_products(filtered_products)
        
        return {
            "query": request.query,
            "stats": {
                "total_scraped": len(request.products),
                "after_filtering": len(filtered_products),
                "product_groups": len(match_result.get("groups", []))
            },
            "ai_powered": filter_result.get("ai_powered", False) or match_result.get("ai_powered", False),
            "filtered_products": filtered_products,
            "product_groups": match_result.get("groups", []),
            "best_deal": filter_result.get("best_deal")
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Starting PriceHunt API Lite v2.0...")
    print("🤖 AI-powered filtering (no scraping)")
    print(f"📱 Running on port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
