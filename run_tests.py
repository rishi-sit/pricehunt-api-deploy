#!/usr/bin/env python3
"""
Quick test runner for PriceHunt API - uses urllib which may bypass SSL issues
"""
import json
import urllib.request
import urllib.error
import ssl
import time
from typing import Dict, List, Any

API_URL = "https://pricehunt-hklm.onrender.com"

# Create SSL context that's more permissive
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

TEST_CASES = [
    {
        "name": "strawberry",
        "query": "strawberry",
        "products": [
            {"name": "Fresh Strawberry 200g", "price": 99, "platform": "Zepto"},
            {"name": "Strawberry Shake 200ml", "price": 45, "platform": "BigBasket"},
            {"name": "Strawberry Jam 200g", "price": 89, "platform": "Amazon"},
            {"name": "American Strawberry 200g Pack", "price": 149, "platform": "Blinkit"},
            {"name": "Strawberry Flavoured Milk 200ml", "price": 35, "platform": "JioMart"},
            {"name": "Fresh Strawberries Premium 250g", "price": 129, "platform": "Instamart"},
            {"name": "Strawberry Ice Cream 500ml", "price": 159, "platform": "BigBasket"},
            {"name": "Driscoll's Strawberries 454g", "price": 299, "platform": "Amazon Fresh"}
        ],
        "expected": ["Fresh Strawberry 200g", "American Strawberry 200g Pack", "Fresh Strawberries Premium 250g", "Driscoll's Strawberries 454g"],
        "best_deal_must_contain": ["strawberry", "strawberries"],
        "best_deal_must_not_contain": ["shake", "jam", "milk", "ice cream"]
    },
    {
        "name": "banana",
        "query": "banana",
        "products": [
            {"name": "Banana Robusta 1kg", "price": 49, "platform": "Zepto"},
            {"name": "Yellaki Banana 12 pcs", "price": 59, "platform": "BigBasket"},
            {"name": "Banana Chips 150g", "price": 45, "platform": "Amazon"},
            {"name": "Banana Cake Slice", "price": 55, "platform": "Blinkit"},
            {"name": "Cavendish Banana 6 pcs", "price": 45, "platform": "Instamart"},
            {"name": "Raw Banana 500g", "price": 35, "platform": "JioMart"},
            {"name": "Banana Wafer 200g", "price": 65, "platform": "BigBasket"},
            {"name": "Ripe Banana Pack 6 pcs", "price": 55, "platform": "Flipkart"}
        ],
        "expected": ["Banana Robusta 1kg", "Yellaki Banana 12 pcs", "Cavendish Banana 6 pcs", "Raw Banana 500g", "Ripe Banana Pack 6 pcs"],
        "best_deal_must_contain": ["banana"],
        "best_deal_must_not_contain": ["chips", "cake", "wafer"]
    },
    {
        "name": "milk",
        "query": "milk",
        "products": [
            {"name": "Amul Taaza Toned Milk 500ml", "price": 28, "platform": "Zepto"},
            {"name": "Mother Dairy Full Cream Milk 1L", "price": 68, "platform": "BigBasket"},
            {"name": "Cadbury Dairy Milk Chocolate 50g", "price": 50, "platform": "Blinkit"},
            {"name": "Nestle Milkshake Strawberry 180ml", "price": 35, "platform": "JioMart"},
            {"name": "Nandini Milk 500ml", "price": 25, "platform": "Instamart"},
            {"name": "Milk Bikis Biscuit 100g", "price": 20, "platform": "Amazon"},
            {"name": "Heritage Slim Milk 500ml", "price": 30, "platform": "BigBasket"},
            {"name": "Milkmaid Condensed Milk 400g", "price": 99, "platform": "Amazon"}
        ],
        "expected": ["Amul Taaza Toned Milk 500ml", "Mother Dairy Full Cream Milk 1L", "Nandini Milk 500ml", "Heritage Slim Milk 500ml"],
        "best_deal_must_contain": ["milk"],
        "best_deal_must_not_contain": ["chocolate", "milkshake", "bikis", "biscuit"]
    },
    {
        "name": "apple",
        "query": "apple",
        "products": [
            {"name": "Fresh Apple Red Delicious 1kg", "price": 180, "platform": "BigBasket"},
            {"name": "Apple iPhone 15 Case", "price": 999, "platform": "Amazon"},
            {"name": "Real Apple Juice 1L", "price": 99, "platform": "Zepto"},
            {"name": "Organic Green Apple 500g", "price": 120, "platform": "JioMart"},
            {"name": "Pineapple Fresh 1kg", "price": 90, "platform": "Instamart"},
            {"name": "Shimla Apple Premium 1kg", "price": 200, "platform": "Blinkit"},
            {"name": "Apple Cider Vinegar 500ml", "price": 250, "platform": "BigBasket"},
            {"name": "Washington Apple 4pcs", "price": 160, "platform": "Zepto"}
        ],
        "expected": ["Fresh Apple Red Delicious 1kg", "Organic Green Apple 500g", "Shimla Apple Premium 1kg", "Washington Apple 4pcs"],
        "best_deal_must_contain": ["apple"],
        "best_deal_must_not_contain": ["juice", "iphone", "pineapple", "vinegar", "cider"]
    },
    {
        "name": "grape",
        "query": "grape",
        "products": [
            {"name": "Fresh Grapes Green 500g", "price": 80, "platform": "BigBasket"},
            {"name": "Grapefruit Fresh 1kg", "price": 120, "platform": "Zepto"},
            {"name": "Red Grapes Seedless 500g", "price": 90, "platform": "Amazon"},
            {"name": "Grape Juice 1L", "price": 99, "platform": "JioMart"},
            {"name": "Black Grapes 500g", "price": 100, "platform": "Instamart"},
            {"name": "Grapeseed Oil 500ml", "price": 250, "platform": "Amazon"}
        ],
        "expected": ["Fresh Grapes Green 500g", "Red Grapes Seedless 500g", "Black Grapes 500g"],
        "best_deal_must_contain": ["grape"],
        "best_deal_must_not_contain": ["grapefruit", "grapeseed", "juice", "oil"]
    },
    {
        "name": "tomato",
        "query": "tomato",
        "products": [
            {"name": "Fresh Tomato Local 1kg", "price": 40, "platform": "Zepto"},
            {"name": "Tomato Ketchup 500g", "price": 99, "platform": "Amazon"},
            {"name": "Kissan Tomato Sauce 500g", "price": 89, "platform": "BigBasket"},
            {"name": "Organic Tomato 500g", "price": 60, "platform": "JioMart"},
            {"name": "Cherry Tomato 200g", "price": 80, "platform": "Instamart"},
            {"name": "Tomato Puree 200g", "price": 45, "platform": "Blinkit"},
            {"name": "Hybrid Tomato 1kg", "price": 55, "platform": "BigBasket"}
        ],
        "expected": ["Fresh Tomato Local 1kg", "Organic Tomato 500g", "Cherry Tomato 200g", "Hybrid Tomato 1kg"],
        "best_deal_must_contain": ["tomato"],
        "best_deal_must_not_contain": ["ketchup", "sauce", "puree"]
    },
    {
        "name": "eggs",
        "query": "eggs",
        "products": [
            {"name": "Farm Fresh Eggs 12pcs", "price": 90, "platform": "BigBasket"},
            {"name": "Country Eggs 6pcs", "price": 65, "platform": "Zepto"},
            {"name": "Eggless Mayo 250g", "price": 60, "platform": "Blinkit"},
            {"name": "Eggless Cake Mix 500g", "price": 80, "platform": "Amazon"},
            {"name": "Brown Eggs 6pcs", "price": 72, "platform": "Instamart"},
            {"name": "Omega-3 Eggs 6pcs", "price": 85, "platform": "JioMart"}
        ],
        "expected": ["Farm Fresh Eggs 12pcs", "Country Eggs 6pcs", "Brown Eggs 6pcs", "Omega-3 Eggs 6pcs"],
        "best_deal_must_contain": ["eggs", "egg"],
        "best_deal_must_not_contain": ["eggless"]
    },
    {
        "name": "mango",
        "query": "mango",
        "products": [
            {"name": "Alphonso Mango 1kg", "price": 450, "platform": "BigBasket"},
            {"name": "Maaza Mango Drink 600ml", "price": 45, "platform": "Zepto"},
            {"name": "Mango Pickle 200g", "price": 65, "platform": "Amazon"},
            {"name": "Kesar Mango 1kg", "price": 350, "platform": "JioMart"},
            {"name": "Raw Mango 500g", "price": 60, "platform": "Instamart"},
            {"name": "Mango Shake Powder 200g", "price": 99, "platform": "Blinkit"},
            {"name": "Badami Mango 1kg", "price": 280, "platform": "BigBasket"}
        ],
        "expected": ["Alphonso Mango 1kg", "Kesar Mango 1kg", "Raw Mango 500g", "Badami Mango 1kg"],
        "best_deal_must_contain": ["mango"],
        "best_deal_must_not_contain": ["drink", "pickle", "shake", "powder", "maaza"]
    }
]


def make_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make HTTP request to API"""
    url = f"{API_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    
    if data:
        req = urllib.request.Request(
            url, 
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method=method
        )
    else:
        req = urllib.request.Request(url, headers=headers, method=method)
    
    try:
        with urllib.request.urlopen(req, timeout=60, context=ssl_context) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"error": f"URL Error: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def run_test(test_case: Dict) -> Dict:
    """Run a single test case"""
    start = time.time()
    
    result = make_request("/api/smart-search", "POST", {
        "query": test_case["query"],
        "products": test_case["products"],
        "strict_mode": True
    })
    
    latency_ms = int((time.time() - start) * 1000)
    
    if "error" in result:
        return {
            "name": test_case["name"],
            "error": result["error"],
            "latency_ms": latency_ms
        }
    
    # Get results
    relevant = result.get("results", [])
    predicted = set(p.get("name", "").lower().strip() for p in relevant)
    expected = set(n.lower().strip() for n in test_case["expected"])
    all_products = set(p.get("name", "").lower().strip() for p in test_case["products"])
    
    # Calculate metrics
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    tn = len((all_products - predicted) & (all_products - expected))
    
    accuracy = (tp + tn) / len(all_products) if all_products else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Check best deal
    best_deal = result.get("best_deal", {})
    best_deal_name = (best_deal.get("name", "") or "").lower()
    
    contains_ok = any(t.lower() in best_deal_name for t in test_case["best_deal_must_contain"])
    excludes_ok = not any(t.lower() in best_deal_name for t in test_case["best_deal_must_not_contain"])
    best_deal_correct = contains_ok and excludes_ok if best_deal_name else False
    
    return {
        "name": test_case["name"],
        "query": test_case["query"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "best_deal_correct": best_deal_correct,
        "best_deal_name": best_deal.get("name", "None"),
        "best_deal_price": best_deal.get("price", 0),
        "ai_powered": result.get("ai_powered", False),
        "ai_model": result.get("ai_meta", {}).get("model", "unknown"),
        "ai_provider": result.get("ai_meta", {}).get("provider", "unknown"),
        "latency_ms": latency_ms,
        "predicted": [p.get("name") for p in relevant],
        "expected": test_case["expected"]
    }


def main():
    print("=" * 70)
    print("  PriceHunt AI Accuracy Test Suite")
    print("=" * 70)
    print(f"API: {API_URL}")
    print()
    
    # Check API health
    print("🔍 Checking API health...")
    health = make_request("/")
    if "error" in health:
        print(f"❌ API Error: {health['error']}")
        return
    
    print(f"✅ API: {health.get('name', 'Unknown')} v{health.get('version', '?')}")
    print(f"   AI Enabled: {health.get('ai_enabled', False)}")
    print()
    
    # Run tests
    print("Running tests...")
    print("-" * 70)
    
    results = []
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] Testing '{tc['name']}'...", end=" ", flush=True)
        result = run_test(tc)
        results.append(result)
        
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            bd_status = "✅" if result["best_deal_correct"] else "❌"
            acc_pct = result["accuracy"] * 100
            status = "✅" if acc_pct >= 80 else "⚠️" if acc_pct >= 60 else "❌"
            print(f"{status} Acc={acc_pct:.0f}% Prec={result['precision']*100:.0f}% Rec={result['recall']*100:.0f}% BestDeal={bd_status} ({result['latency_ms']}ms)")
        
        time.sleep(0.5)  # Rate limiting
    
    # Summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("❌ No successful tests!")
        return
    
    avg_accuracy = sum(r["accuracy"] for r in valid) / len(valid)
    avg_precision = sum(r["precision"] for r in valid) / len(valid)
    avg_recall = sum(r["recall"] for r in valid) / len(valid)
    best_deal_acc = sum(1 for r in valid if r["best_deal_correct"]) / len(valid)
    avg_latency = sum(r["latency_ms"] for r in valid) / len(valid)
    
    print(f"📊 Tests Run: {len(results)} ({len(valid)} successful)")
    print(f"📈 Average Accuracy: {avg_accuracy*100:.1f}%")
    print(f"📈 Average Precision: {avg_precision*100:.1f}%")
    print(f"📈 Average Recall: {avg_recall*100:.1f}%")
    print(f"🎯 Best Deal Accuracy: {best_deal_acc*100:.1f}%")
    print(f"⏱️  Average Latency: {avg_latency:.0f}ms")
    
    # AI Provider info
    if valid:
        ai_info = valid[0]
        print(f"\n🤖 AI Provider: {ai_info.get('ai_provider', 'unknown')}")
        print(f"🤖 AI Model: {ai_info.get('ai_model', 'unknown')}")
    
    # Problem cases
    problems = [r for r in valid if r["accuracy"] < 0.8 or not r["best_deal_correct"]]
    if problems:
        print("\n⚠️  Problem Cases:")
        for p in problems:
            print(f"   - {p['name']}: Acc={p['accuracy']*100:.0f}% BestDeal={'✅' if p['best_deal_correct'] else '❌ '+p['best_deal_name']}")
            if p["fp"] > 0:
                wrong = set(n.lower() for n in p["predicted"]) - set(n.lower() for n in p["expected"])
                print(f"     False Positives: {list(wrong)[:3]}")
            if p["fn"] > 0:
                missed = set(n.lower() for n in p["expected"]) - set(n.lower() for n in p["predicted"])
                print(f"     Missed: {list(missed)[:3]}")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
