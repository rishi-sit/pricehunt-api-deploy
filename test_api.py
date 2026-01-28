#!/usr/bin/env python3
"""
Test script for PriceHunt API (without Gemini - tests fallback mode)
"""
import json

# Mock the google.generativeai module to avoid import errors
import sys
from unittest.mock import MagicMock

# Create mock module
mock_genai = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = mock_genai

# Now import FastAPI test client
from fastapi.testclient import TestClient

# Import the app
from api_server import app

client = TestClient(app)


def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)
    
    response = client.get("/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ PASSED")


def test_root():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("TEST: Root Endpoint")
    print("="*60)
    
    response = client.get("/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert "PriceHunt API" in response.json()["message"]
    print("✅ PASSED")


def test_smart_search():
    """Test smart search endpoint with sample data"""
    print("\n" + "="*60)
    print("TEST: Smart Search (milk query)")
    print("="*60)
    
    # Sample products simulating Android scraping
    test_products = [
        {"name": "Amul Toned Milk 500ml", "price": 28, "platform": "Zepto", "available": True},
        {"name": "Mother Dairy Full Cream Milk 1L", "price": 68, "platform": "BigBasket", "available": True},
        {"name": "Milkmaid Condensed Milk 400g", "price": 99, "platform": "Amazon Fresh", "available": True},
        {"name": "Cadbury Dairy Milk Chocolate 50g", "price": 50, "platform": "Blinkit", "available": True},
        {"name": "Nestle Milkshake Strawberry 180ml", "price": 35, "platform": "JioMart", "available": True},
        {"name": "Amul Taaza Toned Milk 500ml", "price": 27, "platform": "Flipkart Minutes", "available": True},
        {"name": "Nandini Milk 500ml Tetra Pack", "price": 25, "platform": "Instamart", "available": True},
    ]
    
    request_body = {
        "query": "milk",
        "products": test_products,
        "pincode": "560001",
        "strict_mode": True
    }
    
    print(f"Request: POST /api/smart-search")
    print(f"Query: 'milk'")
    print(f"Input products: {len(test_products)}")
    
    response = client.post("/api/smart-search", json=request_body)
    print(f"\nStatus: {response.status_code}")
    
    data = response.json()
    print(f"\n--- Response Summary ---")
    print(f"AI Powered: {data.get('ai_powered', 'N/A')}")
    print(f"Relevant products: {data['stats']['total_relevant']}")
    print(f"Filtered out: {data['stats']['total_filtered']}")
    
    if data.get('best_deal'):
        print(f"\nBest Deal: {data['best_deal']['name']} @ ₹{data['best_deal']['price']} ({data['best_deal']['platform']})")
    
    print(f"\n--- Relevant Products ---")
    for p in data.get('results', [])[:5]:
        score = p.get('relevance_score', 'N/A')
        print(f"  ✅ [{score}] {p['name'][:40]} - ₹{p['price']} ({p['platform']})")
    
    print(f"\n--- Filtered Out ---")
    for p in data.get('filtered_out', [])[:5]:
        print(f"  ❌ {p['name'][:40]} - {p['filter_reason']}")
    
    assert response.status_code == 200
    # In fallback mode, some filtering should still happen
    print("\n✅ PASSED")


def test_smart_search_and_match():
    """Test combined smart search + product matching"""
    print("\n" + "="*60)
    print("TEST: Smart Search & Match (milk query)")
    print("="*60)
    
    # Products from multiple platforms - some are same product different platform
    test_products = [
        {"name": "Amul Taaza Toned Milk 500ml", "price": 28, "platform": "Zepto", "available": True},
        {"name": "Amul Taaza 500ml Tetra Pack", "price": 30, "platform": "BigBasket", "available": True},
        {"name": "Amul Taaza Homogenised Toned Milk 500 ml", "price": 27, "platform": "JioMart", "available": True},
        {"name": "Mother Dairy Full Cream Milk 1L", "price": 68, "platform": "Zepto", "available": True},
        {"name": "Mother Dairy Full Cream 1 Litre", "price": 72, "platform": "BigBasket", "available": True},
        {"name": "Milkshake Chocolate 200ml", "price": 45, "platform": "Blinkit", "available": True},
        {"name": "Cadbury Dairy Milk Silk", "price": 99, "platform": "Amazon", "available": True},
    ]
    
    request_body = {
        "query": "milk",
        "products": test_products,
        "pincode": "560001",
        "strict_mode": True
    }
    
    print(f"Request: POST /api/smart-search-and-match")
    print(f"Query: 'milk'")
    print(f"Input products: {len(test_products)}")
    
    response = client.post("/api/smart-search-and-match", json=request_body)
    print(f"\nStatus: {response.status_code}")
    
    data = response.json()
    print(f"\n--- Response Summary ---")
    print(f"AI Powered: {data.get('ai_powered', 'N/A')}")
    print(f"Relevant products: {data['stats']['relevant_products']}")
    print(f"Filtered out: {data['stats']['filtered_products']}")
    print(f"Product groups: {data['stats']['product_groups']}")
    
    if data.get('best_deal'):
        print(f"\nBest Deal: {data['best_deal']['name'][:40]} @ ₹{data['best_deal']['price']}")
    
    print(f"\n--- Product Groups (cross-platform matches) ---")
    for group in data.get('product_groups', [])[:3]:
        print(f"\n  📦 {group['canonical_name']}")
        print(f"     Price range: {group['price_range']}")
        if group.get('savings'):
            print(f"     Potential savings: ₹{group['savings']}")
        print(f"     Available at:")
        for p in group['products'][:3]:
            print(f"       - {p['platform']}: ₹{p.get('price', 'N/A')}")
    
    assert response.status_code == 200
    print("\n✅ PASSED")


def test_understand_query():
    """Test query understanding endpoint"""
    print("\n" + "="*60)
    print("TEST: Understand Query")
    print("="*60)
    
    queries = ["milk 500ml", "organic apple", "sunflower oil 1L"]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        response = client.get(f"/api/understand-query?q={query}")
        data = response.json()
        print(f"  AI Powered: {data.get('ai_powered', 'N/A')}")
        understanding = data.get('understanding', {})
        print(f"  Product type: {understanding.get('product_type', 'N/A')}")
        print(f"  Quantity: {understanding.get('quantity', 'N/A')}")
        print(f"  Category: {understanding.get('category', 'N/A')}")
    
    print("\n✅ PASSED")


def test_match_products():
    """Test product matching endpoint"""
    print("\n" + "="*60)
    print("TEST: Match Products")
    print("="*60)
    
    # Same products from different platforms
    test_products = [
        {"name": "Fortune Sunflower Oil 1L", "price": 145, "platform": "Zepto", "available": True},
        {"name": "Fortune Sunlite Refined Sunflower Oil 1 Litre", "price": 150, "platform": "BigBasket", "available": True},
        {"name": "Fortune Sunflower Refined Oil 1L Pouch", "price": 142, "platform": "JioMart", "available": True},
        {"name": "Saffola Gold Oil 1L", "price": 189, "platform": "Zepto", "available": True},
        {"name": "Saffola Gold Blended Oil 1 Litre", "price": 195, "platform": "Amazon Fresh", "available": True},
    ]
    
    request_body = {"products": test_products}
    
    print(f"Request: POST /api/match-products")
    print(f"Input products: {len(test_products)}")
    
    response = client.post("/api/match-products", json=request_body)
    print(f"\nStatus: {response.status_code}")
    
    data = response.json()
    print(f"\n--- Response Summary ---")
    print(f"AI Powered: {data.get('ai_powered', 'N/A')}")
    print(f"Product groups: {data['stats']['total_groups']}")
    print(f"Matched products: {data['stats']['total_matched']}")
    print(f"Unmatched: {data['stats']['total_unmatched']}")
    
    print(f"\n--- Matched Groups ---")
    for group in data.get('product_groups', []):
        print(f"\n  📦 {group['canonical_name']}")
        print(f"     Brand: {group.get('brand', 'N/A')}")
        print(f"     Quantity: {group.get('quantity', 'N/A')}")
        print(f"     Price range: {group['price_range']}")
        if group.get('best_deal'):
            print(f"     Best deal: {group['best_deal']['platform']} @ ₹{group['best_deal']['price']}")
    
    assert response.status_code == 200
    print("\n✅ PASSED")


if __name__ == "__main__":
    print("\n" + "🧪"*30)
    print("   PRICEHUNT API E2E TEST (Fallback Mode - No Gemini)")
    print("🧪"*30)
    
    try:
        test_health()
        test_root()
        test_smart_search()
        test_smart_search_and_match()
        test_understand_query()
        test_match_products()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nNote: Tests ran in FALLBACK mode (without Gemini AI)")
        print("To test with AI, set GEMINI_API_KEY environment variable")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
