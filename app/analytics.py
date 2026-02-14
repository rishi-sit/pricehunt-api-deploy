"""
Analytics Module for PriceHunt Scraping Dashboard.

Tracks scraping metrics with device_id as primary key:
- Scrape source (device vs server)
- HTML response size (KB)
- Products scraped per platform
- Relevant products suggested by AI model
"""

import sqlite3
import os
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from contextlib import contextmanager


# ============================================================================
# Database Configuration
# ============================================================================

# Use environment variable for persistence, fallback to local file
DB_PATH = os.environ.get("ANALYTICS_DB_PATH", "pricehunt_analytics.db")


def get_db_connection():
    """Get SQLite connection with proper settings."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database():
    """Initialize the analytics database with required tables."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Main scrape logs table (from Android app)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scrape_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                search_query TEXT NOT NULL,
                platform TEXT NOT NULL,
                scrape_source TEXT NOT NULL,  -- 'device', 'ai_fallback', 'playwright', 'cache'
                html_size_kb REAL DEFAULT 0,
                products_scraped INTEGER DEFAULT 0,
                relevant_products INTEGER DEFAULT 0,
                ai_model TEXT,  -- 'groq-mistral', 'gemini', 'playwright', null
                success BOOLEAN DEFAULT 1,
                error_message TEXT,
                latency_ms INTEGER DEFAULT 0,
                pincode TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # NEW: Backend AI processing logs (server-side analytics)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT,
                search_query TEXT NOT NULL,
                endpoint TEXT NOT NULL,  -- 'ai-extract', 'smart-search', 'match-products'
                ai_provider TEXT NOT NULL,  -- 'groq', 'mistral', 'gemini'
                ai_model TEXT NOT NULL,  -- 'mixtral-8x7b', 'gemini-pro', etc.
                fallback_reason TEXT,  -- why fallback was used
                input_products INTEGER DEFAULT 0,
                output_products INTEGER DEFAULT 0,
                filtered_out INTEGER DEFAULT 0,
                tokens_used INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                success BOOLEAN DEFAULT 1,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_device_date 
            ON scrape_logs(device_id, date(created_at))
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_platform 
            ON scrape_logs(platform)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON scrape_logs(created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ai_logs_date 
            ON ai_processing_logs(date(created_at))
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ai_logs_provider 
            ON ai_processing_logs(ai_provider)
        """)
        
        # Summary table for quick dashboard queries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                date DATE NOT NULL,
                platform TEXT NOT NULL,
                total_searches INTEGER DEFAULT 0,
                device_scrapes INTEGER DEFAULT 0,
                ai_scrapes INTEGER DEFAULT 0,
                playwright_scrapes INTEGER DEFAULT 0,
                cache_hits INTEGER DEFAULT 0,
                total_products_scraped INTEGER DEFAULT 0,
                total_relevant_products INTEGER DEFAULT 0,
                avg_html_size_kb REAL DEFAULT 0,
                avg_latency_ms INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0,
                UNIQUE(device_id, date, platform)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_device_date 
            ON daily_summary(device_id, date)
        """)
        
        conn.commit()
        print(f"[Analytics] Database initialized at {DB_PATH}")


# ============================================================================
# Pydantic Models
# ============================================================================

class ScrapeLogRequest(BaseModel):
    """Request model for logging a scrape event."""
    device_id: str
    search_query: str
    platform: str
    scrape_source: str  # 'device', 'ai_fallback', 'playwright', 'cache'
    html_size_kb: float = 0.0
    products_scraped: int = 0
    relevant_products: int = 0
    ai_model: Optional[str] = None  # 'groq-mistral', 'gemini', 'playwright'
    success: bool = True
    error_message: Optional[str] = None
    latency_ms: int = 0
    pincode: Optional[str] = None


class ScrapeLogResponse(BaseModel):
    """Response model for a logged scrape event."""
    id: int
    device_id: str
    search_query: str
    platform: str
    scrape_source: str
    html_size_kb: float
    products_scraped: int
    relevant_products: int
    ai_model: Optional[str]
    success: bool
    error_message: Optional[str]
    latency_ms: int
    pincode: Optional[str]
    created_at: str


class AIProcessingLogRequest(BaseModel):
    """Request model for logging backend AI processing."""
    device_id: Optional[str] = None
    search_query: str
    endpoint: str  # 'ai-extract', 'smart-search', 'match-products'
    ai_provider: str  # 'groq', 'mistral', 'gemini'
    ai_model: str  # 'mixtral-8x7b', 'gemini-pro', etc.
    fallback_reason: Optional[str] = None
    input_products: int = 0
    output_products: int = 0
    filtered_out: int = 0
    tokens_used: int = 0
    latency_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None


class AIProcessingStats(BaseModel):
    """Statistics for AI processing."""
    provider: str
    model: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: int
    total_input_products: int
    total_output_products: int
    total_filtered_out: int
    total_tokens_used: int


class PlatformStats(BaseModel):
    """Statistics for a single platform."""
    platform: str
    total_searches: int
    device_scrapes: int
    ai_scrapes: int
    playwright_scrapes: int
    cache_hits: int
    total_products_scraped: int
    total_relevant_products: int
    avg_html_size_kb: float
    avg_latency_ms: int
    success_rate: float


class DashboardResponse(BaseModel):
    """Response model for dashboard data."""
    device_id: str
    date: str
    total_searches: int
    platform_stats: List[PlatformStats]
    overall_success_rate: float
    total_products_scraped: int
    total_relevant_products: int
    ai_model_usage: Dict[str, int]
    scrape_source_breakdown: Dict[str, int]
    # NEW: Backend AI processing stats
    ai_processing_stats: Optional[List[AIProcessingStats]] = None


class DashboardQueryRequest(BaseModel):
    """Request model for querying dashboard data."""
    device_id: str
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD

class BulkLogRequest(BaseModel):
    """Request for logging multiple scrape events at once (batch)."""
    logs: List[ScrapeLogRequest]


# ============================================================================
# Analytics Functions
# ============================================================================

def log_scrape_event(log: ScrapeLogRequest) -> int:
    """Log a single scrape event to the database."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scrape_logs (
                device_id, search_query, platform, scrape_source,
                html_size_kb, products_scraped, relevant_products,
                ai_model, success, error_message, latency_ms, pincode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log.device_id,
            log.search_query,
            log.platform,
            log.scrape_source,
            log.html_size_kb,
            log.products_scraped,
            log.relevant_products,
            log.ai_model,
            log.success,
            log.error_message,
            log.latency_ms,
            log.pincode
        ))
        return cursor.lastrowid


def log_bulk_events(logs: List[ScrapeLogRequest]) -> List[int]:
    """Log multiple scrape events efficiently."""
    ids = []
    with get_db() as conn:
        cursor = conn.cursor()
        for log in logs:
            cursor.execute("""
                INSERT INTO scrape_logs (
                    device_id, search_query, platform, scrape_source,
                    html_size_kb, products_scraped, relevant_products,
                    ai_model, success, error_message, latency_ms, pincode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log.device_id,
                log.search_query,
                log.platform,
                log.scrape_source,
                log.html_size_kb,
                log.products_scraped,
                log.relevant_products,
                log.ai_model,
                log.success,
                log.error_message,
                log.latency_ms,
                log.pincode
            ))
            ids.append(cursor.lastrowid)
    return ids


def get_dashboard_data(
    device_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> DashboardResponse:
    """Get dashboard data for a device within a date range."""
    
    # Default to today if no dates specified
    if not start_date:
        start_date = date.today().isoformat()
    if not end_date:
        end_date = date.today().isoformat()
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get platform-wise stats
        cursor.execute("""
            SELECT 
                platform,
                COUNT(*) as total_searches,
                SUM(CASE WHEN scrape_source = 'device' THEN 1 ELSE 0 END) as device_scrapes,
                SUM(CASE WHEN scrape_source = 'ai_fallback' THEN 1 ELSE 0 END) as ai_scrapes,
                SUM(CASE WHEN scrape_source = 'playwright' THEN 1 ELSE 0 END) as playwright_scrapes,
                SUM(CASE WHEN scrape_source = 'cache' THEN 1 ELSE 0 END) as cache_hits,
                SUM(products_scraped) as total_products_scraped,
                SUM(relevant_products) as total_relevant_products,
                AVG(html_size_kb) as avg_html_size_kb,
                AVG(latency_ms) as avg_latency_ms,
                AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) * 100 as success_rate
            FROM scrape_logs
            WHERE device_id = ?
              AND date(created_at) >= date(?)
              AND date(created_at) <= date(?)
            GROUP BY platform
            ORDER BY total_searches DESC
        """, (device_id, start_date, end_date))
        
        platform_rows = cursor.fetchall()
        platform_stats = []
        total_searches = 0
        total_products = 0
        total_relevant = 0
        
        for row in platform_rows:
            stat = PlatformStats(
                platform=row['platform'],
                total_searches=row['total_searches'] or 0,
                device_scrapes=row['device_scrapes'] or 0,
                ai_scrapes=row['ai_scrapes'] or 0,
                playwright_scrapes=row['playwright_scrapes'] or 0,
                cache_hits=row['cache_hits'] or 0,
                total_products_scraped=row['total_products_scraped'] or 0,
                total_relevant_products=row['total_relevant_products'] or 0,
                avg_html_size_kb=round(row['avg_html_size_kb'] or 0, 2),
                avg_latency_ms=int(row['avg_latency_ms'] or 0),
                success_rate=round(row['success_rate'] or 0, 1)
            )
            platform_stats.append(stat)
            total_searches += stat.total_searches
            total_products += stat.total_products_scraped
            total_relevant += stat.total_relevant_products
        
        # Get AI model usage breakdown
        cursor.execute("""
            SELECT 
                COALESCE(ai_model, 'none') as model,
                COUNT(*) as count
            FROM scrape_logs
            WHERE device_id = ?
              AND date(created_at) >= date(?)
              AND date(created_at) <= date(?)
            GROUP BY ai_model
        """, (device_id, start_date, end_date))
        
        ai_model_usage = {row['model']: row['count'] for row in cursor.fetchall()}
        
        # Get scrape source breakdown
        cursor.execute("""
            SELECT 
                scrape_source,
                COUNT(*) as count
            FROM scrape_logs
            WHERE device_id = ?
              AND date(created_at) >= date(?)
              AND date(created_at) <= date(?)
            GROUP BY scrape_source
        """, (device_id, start_date, end_date))
        
        scrape_source_breakdown = {row['scrape_source']: row['count'] for row in cursor.fetchall()}
        
        # Calculate overall success rate
        cursor.execute("""
            SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) * 100 as overall_success_rate
            FROM scrape_logs
            WHERE device_id = ?
              AND date(created_at) >= date(?)
              AND date(created_at) <= date(?)
        """, (device_id, start_date, end_date))
        
        overall_success_rate = cursor.fetchone()['overall_success_rate'] or 0.0
        
        return DashboardResponse(
            device_id=device_id,
            date=f"{start_date} to {end_date}" if start_date != end_date else start_date,
            total_searches=total_searches,
            platform_stats=platform_stats,
            overall_success_rate=round(overall_success_rate, 1),
            total_products_scraped=total_products,
            total_relevant_products=total_relevant,
            ai_model_usage=ai_model_usage,
            scrape_source_breakdown=scrape_source_breakdown
        )


def get_recent_logs(
    device_id: str,
    limit: int = 100,
    platform: Optional[str] = None
) -> List[ScrapeLogResponse]:
    """Get recent scrape logs for a device."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM scrape_logs
            WHERE device_id = ?
        """
        params = [device_id]
        
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        return [
            ScrapeLogResponse(
                id=row['id'],
                device_id=row['device_id'],
                search_query=row['search_query'],
                platform=row['platform'],
                scrape_source=row['scrape_source'],
                html_size_kb=row['html_size_kb'] or 0.0,
                products_scraped=row['products_scraped'] or 0,
                relevant_products=row['relevant_products'] or 0,
                ai_model=row['ai_model'],
                success=bool(row['success']),
                error_message=row['error_message'],
                latency_ms=row['latency_ms'] or 0,
                pincode=row['pincode'],
                created_at=row['created_at']
            )
            for row in cursor.fetchall()
        ]


def get_all_devices() -> List[Dict[str, Any]]:
    """Get list of all devices with their last activity."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                device_id,
                COUNT(*) as total_logs,
                MAX(created_at) as last_activity,
                COUNT(DISTINCT date(created_at)) as active_days
            FROM scrape_logs
            GROUP BY device_id
            ORDER BY last_activity DESC
        """)
        
        return [
            {
                "device_id": row['device_id'],
                "total_logs": row['total_logs'],
                "last_activity": row['last_activity'],
                "active_days": row['active_days']
            }
            for row in cursor.fetchall()
        ]


def log_ai_processing(log: AIProcessingLogRequest) -> int:
    """Log AI processing event from backend."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ai_processing_logs (
                device_id, search_query, platform, ai_provider, ai_model,
                input_html_size_kb, products_found, products_filtered,
                latency_ms, fallback_reason, success
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log.device_id,
            log.search_query,
            log.platform,
            log.ai_provider,
            log.ai_model,
            log.input_html_size_kb,
            log.products_found,
            log.products_filtered,
            log.latency_ms,
            log.fallback_reason,
            log.success
        ))
        return cursor.lastrowid


def get_ai_processing_stats(
    device_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict:
    """Get AI processing statistics for a device."""
    
    if not start_date:
        start_date = date.today().isoformat()
    if not end_date:
        end_date = date.today().isoformat()
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get total AI processing stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_requests,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                AVG(latency_ms) as avg_latency_ms,
                SUM(products_found) as total_products_found,
                SUM(products_filtered) as total_products_filtered
            FROM ai_processing_logs
            WHERE device_id = ?
              AND date(created_at) BETWEEN ? AND ?
        """, (device_id, start_date, end_date))
        
        totals = cursor.fetchone()
        
        # Get stats by AI provider
        cursor.execute("""
            SELECT 
                ai_provider,
                ai_model,
                COUNT(*) as requests,
                AVG(latency_ms) as avg_latency_ms,
                SUM(products_found) as products_found,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
            FROM ai_processing_logs
            WHERE device_id = ?
              AND date(created_at) BETWEEN ? AND ?
            GROUP BY ai_provider, ai_model
        """, (device_id, start_date, end_date))
        
        provider_stats = [
            {
                "provider": row['ai_provider'],
                "model": row['ai_model'],
                "requests": row['requests'],
                "avg_latency_ms": row['avg_latency_ms'],
                "products_found": row['products_found'],
                "success_rate": (row['success_count'] / row['requests'] * 100) if row['requests'] > 0 else 0
            }
            for row in cursor.fetchall()
        ]
        
        # Get fallback reasons
        cursor.execute("""
            SELECT 
                fallback_reason,
                COUNT(*) as count
            FROM ai_processing_logs
            WHERE device_id = ?
              AND date(created_at) BETWEEN ? AND ?
              AND fallback_reason IS NOT NULL
            GROUP BY fallback_reason
        """, (device_id, start_date, end_date))
        
        fallback_reasons = [
            {"reason": row['fallback_reason'], "count": row['count']}
            for row in cursor.fetchall()
        ]
        
        return {
            "total_requests": totals['total_requests'] or 0,
            "successful": totals['successful'] or 0,
            "failed": totals['failed'] or 0,
            "avg_latency_ms": round(totals['avg_latency_ms'] or 0, 2),
            "total_products_found": totals['total_products_found'] or 0,
            "total_products_filtered": totals['total_products_filtered'] or 0,
            "provider_stats": provider_stats,
            "fallback_reasons": fallback_reasons
        }


def get_combined_dashboard(
    device_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict:
    """Get combined analytics dashboard with both scrape logs and AI processing stats."""
    
    scrape_dashboard = get_dashboard_data(device_id, start_date, end_date)
    ai_stats = get_ai_processing_stats(device_id, start_date, end_date)
    quota_stats = get_ai_quota_stats()  # Add global quota stats
    
    return {
        "device_id": device_id,
        "date_range": {
            "start_date": start_date or date.today().isoformat(),
            "end_date": end_date or date.today().isoformat()
        },
        "scrape_stats": scrape_dashboard.dict() if hasattr(scrape_dashboard, 'dict') else scrape_dashboard,
        "ai_processing_stats": ai_stats,
        "ai_quota_stats": quota_stats
    }


def get_ai_quota_stats() -> Dict:
    """
    Get global AI quota usage stats across ALL devices and ALL time.
    
    Shows total hits per AI provider to help track quota limits:
    - Groq: 6000 requests/day free tier
    - Mistral: 1B tokens/month free tier  
    - Gemini: 60 requests/minute, 1500/day free tier
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get total hits per provider (all time)
        cursor.execute("""
            SELECT 
                ai_provider,
                ai_model,
                COUNT(*) as total_requests,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                AVG(latency_ms) as avg_latency_ms,
                SUM(products_found) as total_products,
                MIN(created_at) as first_request,
                MAX(created_at) as last_request
            FROM ai_processing_logs
            GROUP BY ai_provider, ai_model
            ORDER BY total_requests DESC
        """)
        
        provider_totals = [
            {
                "provider": row['ai_provider'],
                "model": row['ai_model'],
                "total_requests": row['total_requests'],
                "successful": row['successful'],
                "failed": row['failed'],
                "success_rate": round((row['successful'] / row['total_requests'] * 100) if row['total_requests'] > 0 else 0, 1),
                "avg_latency_ms": round(row['avg_latency_ms'] or 0, 1),
                "total_products": row['total_products'] or 0,
                "first_request": row['first_request'],
                "last_request": row['last_request']
            }
            for row in cursor.fetchall()
        ]
        
        # Get today's hits per provider (for daily quota tracking)
        today = date.today().isoformat()
        cursor.execute("""
            SELECT 
                ai_provider,
                COUNT(*) as today_requests,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as today_successful
            FROM ai_processing_logs
            WHERE date(created_at) = ?
            GROUP BY ai_provider
        """, (today,))
        
        today_stats = {
            row['ai_provider']: {
                "today_requests": row['today_requests'],
                "today_successful": row['today_successful']
            }
            for row in cursor.fetchall()
        }
        
        # Get this hour's hits (for rate limiting awareness)
        cursor.execute("""
            SELECT 
                ai_provider,
                COUNT(*) as hour_requests
            FROM ai_processing_logs
            WHERE created_at >= datetime('now', '-1 hour')
            GROUP BY ai_provider
        """)
        
        hour_stats = {
            row['ai_provider']: row['hour_requests']
            for row in cursor.fetchall()
        }
        
        # Get total across all providers
        cursor.execute("""
            SELECT 
                COUNT(*) as total_all_time,
                SUM(CASE WHEN date(created_at) = ? THEN 1 ELSE 0 END) as total_today
            FROM ai_processing_logs
        """, (today,))
        
        totals = cursor.fetchone()
        
        # Known quota limits
        quota_limits = {
            "groq": {"daily": 6000, "description": "6000 requests/day free tier"},
            "mistral": {"daily": 10000, "description": "~10K requests/day (1B tokens/month)"},
            "gemini": {"daily": 1500, "per_minute": 60, "description": "1500/day, 60/minute free tier"}
        }
        
        # Enrich provider totals with quota info
        for provider in provider_totals:
            pname = provider['provider'].lower()
            provider['today_requests'] = today_stats.get(pname, {}).get('today_requests', 0)
            provider['hour_requests'] = hour_stats.get(pname, 0)
            if pname in quota_limits:
                provider['quota_limit'] = quota_limits[pname]
                daily_limit = quota_limits[pname].get('daily', 0)
                today_used = provider['today_requests']
                provider['quota_remaining'] = max(0, daily_limit - today_used)
                provider['quota_percent_used'] = round((today_used / daily_limit * 100) if daily_limit > 0 else 0, 1)
        
        return {
            "total_all_time": totals['total_all_time'] or 0,
            "total_today": totals['total_today'] or 0,
            "providers": provider_totals,
            "quota_limits": quota_limits
        }


# Initialize database on module import
init_database()
