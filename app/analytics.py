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
        
        # Main scrape logs table
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


# Initialize database on module import
init_database()
