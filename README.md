# PriceHunt API v2.0 - AI-Powered Price Comparison

## What's New in v2.0

- **Gemini AI Smart Search** - Filters out irrelevant products (e.g., "milkshake" when searching "milk")
- **Product Matching** - Groups same products across platforms for easy comparison
- **Natural Language Understanding** - Understands search intent

## Quick Start

### 1. Get a Gemini API Key (Free!)

1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy your API key

### 2. Local Development

```bash
cd /Users/r0k02i7/pricehunt/pricehunt-api-deploy

# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Install dependencies
pip install -r api_requirements.txt
playwright install chromium

# Run the server
python api_server.py
```

Server runs at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### 3. Deploy to Render

1. Push to GitHub
2. Go to https://render.com and connect your repo
3. **Important**: Add environment variable in Render dashboard:
   - Key: `GEMINI_API_KEY`
   - Value: Your Gemini API key

```bash
# Push to GitHub
git add .
git commit -m "Add AI-powered smart search"
git push origin main
```

---

## API Endpoints

### Original Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search?q=milk&pincode=560001` | GET | Server-side scraping |
| `/api/platforms` | GET | List supported platforms |

### NEW: AI-Powered Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/smart-search` | POST | Filter products with AI |
| `/api/match-products` | POST | Match products across platforms |
| `/api/smart-search-and-match` | POST | **Recommended**: Filter + Match combined |
| `/api/understand-query?q=milk` | GET | Analyze query intent |
| `/api/health` | GET | Health check with AI status |

---

## Usage Examples

### Smart Search (Recommended)

Send scraped products from Android app:

```bash
curl -X POST "http://localhost:8000/api/smart-search-and-match" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "milk",
    "pincode": "560001",
    "strict_mode": true,
    "products": [
      {"name": "Amul Toned Milk 500ml", "price": 28, "platform": "Zepto"},
      {"name": "Milkmaid Condensed 400g", "price": 99, "platform": "BigBasket"},
      {"name": "Cadbury Dairy Milk Chocolate", "price": 50, "platform": "Amazon"},
      {"name": "Mother Dairy Full Cream Milk 1L", "price": 68, "platform": "JioMart"},
      {"name": "Nestle Milkshake Strawberry", "price": 35, "platform": "Blinkit"}
    ]
  }'
```

**Response:**
```json
{
  "query": "milk",
  "ai_powered": true,
  "product_groups": [
    {
      "canonical_name": "Amul Toned Milk 500ml",
      "products": [
        {"name": "Amul Toned Milk 500ml", "price": 28, "platform": "Zepto"}
      ],
      "best_deal": {"platform": "Zepto", "price": 28}
    }
  ],
  "all_products": [
    {"name": "Amul Toned Milk 500ml", "price": 28, "relevance_score": 95},
    {"name": "Mother Dairy Full Cream Milk 1L", "price": 68, "relevance_score": 92}
  ],
  "filtered_out": [
    {"name": "Milkmaid Condensed 400g", "filter_reason": "Brand contains 'milk' but product is condensed milk"},
    {"name": "Cadbury Dairy Milk Chocolate", "filter_reason": "'Dairy Milk' is brand name, product is chocolate"},
    {"name": "Nestle Milkshake Strawberry", "filter_reason": "Milkshake is a derivative, not fresh milk"}
  ],
  "best_deal": {"name": "Amul Toned Milk 500ml", "price": 28, "platform": "Zepto"},
  "stats": {
    "input_products": 5,
    "relevant_products": 2,
    "filtered_products": 3
  }
}
```

### Understand Query Intent

```bash
curl "http://localhost:8000/api/understand-query?q=organic%20apple%201kg"
```

**Response:**
```json
{
  "query": "organic apple 1kg",
  "ai_powered": true,
  "understanding": {
    "product_type": "apple",
    "quantity": "1kg",
    "category": "fruits",
    "search_terms": ["apple", "organic apple", "fresh apple"],
    "exclude_terms": ["apple juice", "apple cider", "apple pie"]
  }
}
```

---

## Android Integration

### 1. Update `ProductRepository.kt`

Inject `SmartSearchRepository`:

```kotlin
@Singleton
class ProductRepository @Inject constructor(
    // ... existing dependencies
    private val smartSearchRepo: SmartSearchRepository
) {
    // After scraping from all platforms, use smart search
    suspend fun searchWithAI(query: String, pincode: String): SmartSearchResult {
        // 1. Scrape products locally (existing code)
        val scrapedProducts = searchAllPlatforms(query, pincode)
        
        // 2. Send to backend for AI filtering
        return smartSearchRepo.smartSearch(
            query = query,
            scrapedProducts = scrapedProducts,
            pincode = pincode
        )
    }
}
```

### 2. Update `HomeViewModel.kt`

```kotlin
// Use AI-powered search
when (val result = productRepository.searchWithAI(query, pincode)) {
    is SmartSearchResult.Success -> {
        _products.value = result.relevantProducts
        _productGroups.value = result.productGroups
        _bestDeal.value = result.bestDeal
        _aiPowered.value = result.aiPowered
    }
    is SmartSearchResult.Error -> {
        // Fallback to local search
        _products.value = localFilteredProducts
    }
}
```

---

## Cost Estimation

Using **Gemini 1.5 Flash** (cheapest model):

| Metric | Value |
|--------|-------|
| Input tokens per search | ~2,500 (50 products × 50 tokens) |
| Output tokens | ~500 |
| Cost per search | ~$0.0002 |
| **1000 searches** | **~$0.20** |

Practically free for personal/small scale use!

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes (for AI) | Google Gemini API key |
| `PORT` | No | Server port (default: 8000) |

---

## Troubleshooting

### AI features not working?

1. Check if `GEMINI_API_KEY` is set:
   ```bash
   curl http://localhost:8000/api/health
   # Should show "ai_available": true
   ```

2. Test the key directly:
   ```bash
   curl "https://generativelanguage.googleapis.com/v1/models?key=YOUR_KEY"
   ```

### Getting rate limited?

Gemini 1.5 Flash has generous free tier limits:
- 15 requests per minute
- 1 million tokens per minute

For higher usage, consider caching or batching requests.

---

## Architecture

```
[Android App]
     │
     ├── WebView Scrapers (scrape ALL products - no local filtering)
     │
     └── POST /api/smart-search-and-match
              │
              ▼
        [Backend API]
              │
              ├── Gemini AI (filter + understand)
              │
              └── Product Matcher (group by brand/size)
                      │
                      ▼
              [Filtered + Matched Results]
                      │
                      ▼
              [Display in App]
              
        ⚠️ If Backend Fails:
              │
              ▼
        [Local SearchIntelligence Fallback]
              │
              └── Rule-based filtering (works offline)
```

This hybrid approach:
- ✅ Keeps scraping on Android (avoids server-side anti-bot issues)
- ✅ Uses AI for smart filtering (solves the "milk vs milkshake" problem)
- ✅ Matches products for easy comparison
- ✅ Minimal API cost (~$0.0002 per search)
- ✅ **Offline fallback** - Local SearchIntelligence works when backend unavailable
# Force redeploy Sat Feb 14 22:16:25 IST 2026
