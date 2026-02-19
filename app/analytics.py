"""
Analytics Module for PriceHunt Scraping Dashboard.

Tracks scraping metrics with device_id as primary key:
- Scrape source (device vs server)
- HTML response size (KB)
- Products scraped per platform
- Relevant products suggested by AI model

Supports:
- Turso (libSQL) for production persistence
- SQLite fallback for local development
"""

import os
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from contextlib import contextmanager
from urllib.parse import urlparse


# ============================================================================
# Database Configuration - Turso (libSQL) or SQLite fallback
# ============================================================================

# Check for Turso connection
TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")
USE_TURSO = TURSO_DATABASE_URL is not None and TURSO_AUTH_TOKEN is not None

if USE_TURSO:
    import libsql_experimental as libsql
    print(f"[Analytics] Using Turso database: {TURSO_DATABASE_URL}")
else:
    import sqlite3
    DB_PATH = os.environ.get("ANALYTICS_DB_PATH", "pricehunt_analytics.db")
    print(f"[Analytics] Using SQLite database: {DB_PATH}")


import re


# Table schemas for SELECT * resolution (libsql doesn't support cursor.description)
TABLE_SCHEMAS = {
    'search_sessions': [
        'id', 'session_id', 'device_id', 'search_query', 'pincode',
        'started_at', 'completed_at', 'total_latency_ms', 'total_platforms',
        'successful_platforms', 'failed_platforms', 'total_products_found',
        'relevant_products', 'filtered_out_products', 'best_deal_platform',
        'best_deal_product', 'best_deal_price', 'ai_calls_gemini', 'ai_calls_groq',
        'ai_calls_mistral', 'total_ai_calls', 'ai_fallback_count', 'metadata', 'status', 'created_at'
    ],
    'platform_scrape_events': [
        'id', 'session_id', 'platform', 'scrape_source', 'started_at', 'completed_at',
        'latency_ms', 'success', 'error_message', 'products_found', 'products_relevant',
        'html_size_kb', 'endpoint_used', 'metadata', 'created_at'
    ],
    'ai_processing_events': [
        'id', 'session_id', 'platform', 'ai_provider', 'ai_model', 'started_at',
        'completed_at', 'latency_ms', 'input_tokens', 'output_tokens', 'products_extracted',
        'products_relevant', 'success', 'error_message', 'fallback_from', 'metadata', 'created_at'
    ],
    'scrape_logs': [
        'id', 'device_id', 'query', 'platform', 'scrape_source', 'html_size_kb',
        'products_scraped', 'products_relevant', 'latency_ms', 'success', 'error_message', 'created_at'
    ],
    'ai_processing_logs': [
        'id', 'device_id', 'search_query', 'platform', 'endpoint', 'ai_provider',
        'ai_model', 'input_html_size_kb', 'products_found', 'products_filtered',
        'latency_ms', 'fallback_reason', 'success', 'error_message', 'created_at'
    ]
}


def extract_column_names(sql: str, table_schemas: dict = TABLE_SCHEMAS) -> list:
    """
    Extract column names/aliases from a SELECT SQL query.
    Handles: SELECT col1, col2 AS alias, COUNT(*) as cnt, SELECT * FROM table
    Required because libsql doesn't support cursor.description.
    """
    # Remove newlines and extra spaces
    sql = ' '.join(sql.split())
    
    # Handle SELECT * FROM table_name by looking up table schema
    # (libsql doesn't support cursor.description)
    
    # 1. Simple Case: SELECT * FROM table
    star_match = re.search(r'SELECT\s+\*\s+FROM\s+([a-zA-Z0-9_]+)', sql, re.IGNORECASE)
    if star_match:
        table_name = star_match.group(1).lower()
        if table_name in table_schemas:
            return table_schemas[table_name]
    
    # 2. Case with WHERE/ORDER: SELECT * FROM table WHERE ...
    # This regex ensures we capture the table name before any clauses
    star_complex_match = re.search(r'SELECT\s+\*\s+FROM\s+([a-zA-Z0-9_]+)(?:\s+|$)', sql, re.IGNORECASE)
    if star_complex_match:
        table_name = star_complex_match.group(1).lower()
        if table_name in table_schemas:
            return table_schemas[table_name]

    # 3. Explicit columns: SELECT col1, col2 ...
    # Find SELECT ... FROM portion
    match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE)
    if not match:
        # Maybe no FROM (e.g., SELECT 1 as test)
        match = re.search(r'SELECT\s+(.+?)(?:\s*$|\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT)', sql, re.IGNORECASE)
    
    if not match:
        return []
    
    columns_str = match.group(1)
    
    # Split by comma, but not inside parentheses
    columns = []
    depth = 0
    current = ""
    for char in columns_str:
        if char == '(':
            depth += 1
            current += char
        elif char == ')':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            columns.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        columns.append(current.strip())
    
    # Extract name or alias from each column
    names = []
    for col in columns:
        col = col.strip()
        # Check for AS alias (case insensitive)
        as_match = re.search(r'\s+[Aa][Ss]\s+(\w+)\s*$', col)
        if as_match:
            names.append(as_match.group(1))
        else:
            # Take last word (handles "table.column" -> "column")
            # But avoid function names without alias
            if '(' in col and ')' in col:
                # It's a function without alias - generate name
                names.append(f"col{len(names)}")
            else:
                parts = col.replace('.', ' ').split()
                names.append(parts[-1] if parts else f"col{len(names)}")
    
    return names


class SmartCursor:
    """
    Wrapper around cursor that tracks the last SQL query
    and can extract column names for dict conversion.
    Needed because libsql doesn't support cursor.description.
    """
    def __init__(self, cursor):
        self._cursor = cursor
        self._last_sql = None
        self._columns = None
    
    def execute(self, sql, params=None):
        self._last_sql = sql
        self._columns = extract_column_names(sql)
        if params:
            # Turso/libsql requires tuple, not list
            params_tuple = tuple(params) if isinstance(params, list) else params
            return self._cursor.execute(sql, params_tuple)
        return self._cursor.execute(sql)
    
    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)
    
    def fetchall(self):
        rows = self._cursor.fetchall()
        if not rows:
            return []
        return [self._row_to_dict(r) for r in rows]
    
    def _row_to_dict(self, row):
        # If already dict-like
        if hasattr(row, 'keys'):
            return dict(row)
        if hasattr(row, '_fields'):
            return row._asdict()
        
        # Use extracted column names
        if self._columns and len(self._columns) == len(row):
            return dict(zip(self._columns, row))
        
        # Fallback to numeric keys
        return {i: v for i, v in enumerate(row)}
    
    @property
    def lastrowid(self):
        return self._cursor.lastrowid
    
    @property
    def rowcount(self):
        return self._cursor.rowcount
    
    @property
    def description(self):
        return self._cursor.description


def dict_factory(cursor, row):
    """Row factory that returns dicts instead of tuples."""
    if cursor.description:
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
    return row


def get_db_connection(max_retries: int = 3, initial_delay: float = 1.0):
    """Get database connection (Turso or SQLite) with retry logic for transient errors."""
    import time
    
    if USE_TURSO:
        last_error = None
        for attempt in range(max_retries):
            try:
                conn = libsql.connect(database=TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)
                # Try to set row_factory for dict access (may or may not be supported)
                try:
                    conn.row_factory = dict_factory
                except:
                    pass  # libsql may not support row_factory
                return conn
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if attempt < max_retries - 1 and ("stream" in error_str or "hrana" in error_str or "404" in error_str or "connection" in error_str):
                    delay = initial_delay * (2 ** attempt)
                    print(f"[Analytics] Turso connection error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"[Analytics] Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
        raise last_error
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


def get_cursor(conn):
    """Get cursor with proper row factory."""
    # Use SmartCursor for Turso since libsql doesn't support cursor.description
    if USE_TURSO:
        return SmartCursor(conn.cursor())
    # SQLite with Row factory works fine with regular cursor
    return conn.cursor()


def row_to_dict(cursor, row):
    """Convert a row (tuple or Row) to a dict using cursor.description."""
    if row is None:
        return None
    # If already dict-like (sqlite3.Row), convert to dict
    if hasattr(row, 'keys'):
        return dict(row)
    # For Turso/libsql, the row might have column_names or be a Row object
    if hasattr(row, '_fields'):  # namedtuple
        return row._asdict()
    if hasattr(row, 'column_names'):  # Some Row objects
        return dict(zip(row.column_names, row))
    # Check if cursor has description
    if cursor.description:
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    # Last resort: return indexed dict
    return {i: v for i, v in enumerate(row)}


def fetchall_as_dicts(cursor):
    """Fetch all rows as list of dicts."""
    # If using SmartCursor, fetchall() already returns dicts
    if isinstance(cursor, SmartCursor):
        return cursor.fetchall()
    
    # For SQLite with row_factory, rows are already dict-like
    rows = cursor.fetchall()
    if not rows:
        return []
    
    # Check if rows are already dict-like
    first_row = rows[0]
    if hasattr(first_row, 'keys'):
        return [dict(r) for r in rows]
    if hasattr(first_row, '_fields'):  # namedtuple
        return [r._asdict() for r in rows]
    
    # Fallback: numbered keys
    return [{i: v for i, v in enumerate(r)} for r in rows]


def fetchone_as_dict(cursor):
    """Fetch one row as dict."""
    # If using SmartCursor, fetchone() already returns dict
    if isinstance(cursor, SmartCursor):
        result = cursor.fetchone()
        return result if result is not None else {}
    
    row = cursor.fetchone()
    if row is None:
        return {}
    
    # For SQLite with row_factory, row is already dict-like
    if hasattr(row, 'keys'):
        return dict(row)
    if hasattr(row, '_fields'):  # namedtuple
        return row._asdict()
    
    # Fallback: numbered keys
    return {i: v for i, v in enumerate(row)}


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        # libsql may not have rollback
        if hasattr(conn, 'rollback'):
            conn.rollback()
        raise e
    finally:
        # libsql connections don't have close() method
        if hasattr(conn, 'close'):
            conn.close()


def _get_id_column():
    """Return appropriate ID column definition (SQLite syntax for both Turso and SQLite)."""
    return "id INTEGER PRIMARY KEY AUTOINCREMENT"


def init_database(max_retries: int = 5, initial_delay: float = 2.0):
    """
    Initialize the analytics database with required tables.
    Includes retry logic for transient Turso connection errors.
    
    Uses 5 retries with exponential backoff (2s, 4s, 8s, 16s, 32s).
    """
    import time
    
    print(f"[Analytics] Initializing database (max_retries={max_retries}, initial_delay={initial_delay}s)...")
    
    last_error = None
    for attempt in range(max_retries):
        try:
            _init_database_internal()
            print(f"[Analytics] Database initialized successfully" + 
                  (f" after {attempt + 1} attempts" if attempt > 0 else ""))
            return
        except ValueError as e:
            # Turso "stream not found" or similar transient errors
            last_error = e
            error_str = str(e).lower()
            if "stream not found" in error_str or "no runtime" in error_str or "hrana" in error_str or "404" in error_str:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                print(f"[Analytics] Transient Turso error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"[Analytics] Retrying in {delay}s...")
                time.sleep(delay)
            else:
                # Not a transient error, re-raise immediately
                print(f"[Analytics] Non-transient error, not retrying: {e}")
                raise
        except Exception as e:
            # Other errors - retry as well since Turso can be flaky
            last_error = e
            delay = initial_delay * (2 ** attempt)
            print(f"[Analytics] Database init error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
            print(f"[Analytics] Retrying in {delay}s...")
            time.sleep(delay)
    
    # All retries exhausted - DON'T crash, just warn
    print(f"[Analytics] WARNING: Database initialization failed after {max_retries} attempts")
    print(f"[Analytics] Last error: {type(last_error).__name__}: {last_error}")
    print(f"[Analytics] The service will continue but analytics may not be persisted.")


def _init_database_internal():
    """Internal database initialization (called by init_database with retry)."""
    with get_db() as conn:
        cursor = get_cursor(conn)
        
        id_col = _get_id_column()
        
        # ====================================================================
        # NEW: Search Sessions table - links all events for one search
        # ====================================================================
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS search_sessions (
                {id_col},
                session_id TEXT NOT NULL UNIQUE,
                device_id TEXT NOT NULL,
                search_query TEXT NOT NULL,
                pincode TEXT,
                
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                total_latency_ms INTEGER DEFAULT 0,
                
                -- Platform results summary
                total_platforms INTEGER DEFAULT 0,
                successful_platforms INTEGER DEFAULT 0,
                failed_platforms INTEGER DEFAULT 0,
                
                -- Product summary
                total_products_found INTEGER DEFAULT 0,
                relevant_products INTEGER DEFAULT 0,
                filtered_out INTEGER DEFAULT 0,
                
                -- Best deal info
                best_deal_platform TEXT,
                best_deal_product TEXT,
                best_deal_price REAL,
                
                -- Pipeline stages completed
                scraping_completed BOOLEAN DEFAULT 0,
                ai_filtering_completed BOOLEAN DEFAULT 0,
                matching_completed BOOLEAN DEFAULT 0,
                
                -- AI calls summary
                total_ai_calls INTEGER DEFAULT 0,
                ai_tokens_used INTEGER DEFAULT 0,
                
                -- Flexible JSON column for future fields
                metadata TEXT DEFAULT '{{}}',
                
                -- Status
                status TEXT DEFAULT 'in_progress',
                error_message TEXT
            )
        """)
        
        # ====================================================================
        # Platform Scrape Events - per platform within a session
        # ====================================================================
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS platform_scrape_events (
                {id_col},
                session_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                
                scrape_source TEXT NOT NULL,
                scrape_tier INTEGER DEFAULT 1,
                
                html_size_kb REAL DEFAULT 0,
                html_truncated BOOLEAN DEFAULT FALSE,
                
                products_found INTEGER DEFAULT 0,
                relevant_products INTEGER DEFAULT 0,
                
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                latency_ms INTEGER DEFAULT 0,
                
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                
                metadata TEXT DEFAULT '{{}}',
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ====================================================================
        # AI Processing Events - each AI call within a session
        # ====================================================================
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS ai_processing_events (
                {id_col},
                session_id TEXT,
                device_id TEXT,
                
                endpoint TEXT NOT NULL,
                platform TEXT,
                
                ai_provider TEXT NOT NULL,
                ai_model TEXT NOT NULL,
                is_fallback BOOLEAN DEFAULT FALSE,
                fallback_reason TEXT,
                
                input_size_kb REAL DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                products_input INTEGER DEFAULT 0,
                products_output INTEGER DEFAULT 0,
                
                latency_ms INTEGER DEFAULT 0,
                
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT,
                http_status INTEGER,
                
                metadata TEXT DEFAULT '{{}}',
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ====================================================================
        # Legacy tables (keep for backward compatibility)
        # ====================================================================
        
        # Main scrape logs table (from Android app) - LEGACY
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS scrape_logs (
                {id_col},
                session_id TEXT,
                device_id TEXT NOT NULL,
                search_query TEXT NOT NULL,
                platform TEXT NOT NULL,
                scrape_source TEXT NOT NULL,
                html_size_kb REAL DEFAULT 0,
                products_scraped INTEGER DEFAULT 0,
                relevant_products INTEGER DEFAULT 0,
                ai_model TEXT,
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT,
                latency_ms INTEGER DEFAULT 0,
                pincode TEXT,
                metadata TEXT DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Backend AI processing logs - LEGACY
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS ai_processing_logs (
                {id_col},
                session_id TEXT,
                device_id TEXT,
                search_query TEXT NOT NULL,
                platform TEXT,
                endpoint TEXT,
                ai_provider TEXT NOT NULL,
                ai_model TEXT NOT NULL,
                fallback_reason TEXT,
                input_html_size_kb REAL DEFAULT 0,
                products_found INTEGER DEFAULT 0,
                products_filtered INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT,
                metadata TEXT DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ====================================================================
        # Indexes for fast queries (SQLite/Turso compatible)
        # ====================================================================
        
        # Session indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_device ON search_sessions(device_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_date ON search_sessions(date(started_at))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON search_sessions(status)")
        
        # Platform events indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_platform_events_session ON platform_scrape_events(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_platform_events_platform ON platform_scrape_events(platform)")
        
        # AI events indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_events_session ON ai_processing_events(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_events_provider ON ai_processing_events(ai_provider)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_events_endpoint ON ai_processing_events(endpoint)")
        
        # Legacy indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_device_date ON scrape_logs(device_id, date(created_at))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_platform ON scrape_logs(platform)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_logs_provider ON ai_processing_logs(ai_provider)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scrape_session ON scrape_logs(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_logs_session ON ai_processing_logs(session_id)")
        
        # Add session_id column to legacy tables if not exists (migration)
        try:
            cursor.execute("ALTER TABLE scrape_logs ADD COLUMN session_id TEXT")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE scrape_logs ADD COLUMN metadata TEXT DEFAULT '{}'")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE ai_processing_logs ADD COLUMN session_id TEXT")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE ai_processing_logs ADD COLUMN metadata TEXT DEFAULT '{}'")
        except:
            pass
        
        # ====================================================================
        # AI Accuracy Logs
        # ====================================================================
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS ai_accuracy_logs (
                {id_col},
                session_id TEXT,
                device_id TEXT,
                
                search_query TEXT NOT NULL,
                query_category TEXT,
                
                ai_provider TEXT NOT NULL,
                ai_model TEXT NOT NULL,
                
                total_input_products INTEGER DEFAULT 0,
                products_kept INTEGER DEFAULT 0,
                products_filtered INTEGER DEFAULT 0,
                
                high_relevance_count INTEGER DEFAULT 0,
                medium_relevance_count INTEGER DEFAULT 0,
                low_relevance_count INTEGER DEFAULT 0,
                
                best_deal_relevance_score INTEGER,
                best_deal_reason TEXT,
                
                latency_ms INTEGER DEFAULT 0,
                
                success BOOLEAN DEFAULT TRUE,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_accuracy_provider ON ai_accuracy_logs(ai_provider, ai_model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_accuracy_date ON ai_accuracy_logs(date(created_at))")
        
        conn.commit()
        db_type = f"Turso ({TURSO_DATABASE_URL})" if USE_TURSO else f"SQLite ({DB_PATH})"
        print(f"[Analytics] Database initialized: {db_type}")


# ============================================================================
# Pydantic Models - NEW Session-Based
# ============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a new search session."""
    session_id: str  # UUID from Android
    device_id: str
    search_query: str
    pincode: Optional[str] = None
    total_platforms: int = 10
    metadata: Optional[Dict[str, Any]] = None  # {app_version, network_type, etc.}


class UpdateSessionRequest(BaseModel):
    """Request to update session after search completes."""
    session_id: str
    completed_at: Optional[str] = None
    total_latency_ms: Optional[int] = None
    successful_platforms: Optional[int] = None
    failed_platforms: Optional[int] = None
    total_products_found: Optional[int] = None
    relevant_products: Optional[int] = None
    filtered_out: Optional[int] = None
    best_deal_platform: Optional[str] = None
    best_deal_product: Optional[str] = None
    best_deal_price: Optional[float] = None
    scraping_completed: Optional[bool] = None
    ai_filtering_completed: Optional[bool] = None
    matching_completed: Optional[bool] = None
    total_ai_calls: Optional[int] = None
    status: Optional[str] = None  # 'completed', 'failed', 'timeout'
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PlatformScrapeEventRequest(BaseModel):
    """Request to log a platform scrape event."""
    session_id: str
    device_id: str
    platform: str
    scrape_source: str  # 'device', 'ai_fallback', 'playwright', 'cache'
    scrape_tier: int = 1  # 1=device, 2=ai_fallback, 3=playwright
    html_size_kb: float = 0.0
    html_truncated: bool = False
    products_found: int = 0
    relevant_products: int = 0
    latency_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class AIProcessingEventRequest(BaseModel):
    """Request to log an AI processing event."""
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    endpoint: str  # 'ai-extract', 'smart-search', 'match-products', 'filter'
    platform: Optional[str] = None
    ai_provider: str  # 'groq', 'mistral', 'gemini'
    ai_model: str
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    input_size_kb: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    products_input: int = 0
    products_output: int = 0
    latency_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None
    http_status: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionDetailResponse(BaseModel):
    """Full session detail with all events."""
    session_id: str
    device_id: str
    search_query: str
    pincode: Optional[str]
    started_at: str
    completed_at: Optional[str]
    total_latency_ms: int
    status: str
    
    # Summary
    total_platforms: int
    successful_platforms: int
    failed_platforms: int
    total_products_found: int
    relevant_products: int
    
    # Best deal
    best_deal_platform: Optional[str]
    best_deal_product: Optional[str]
    best_deal_price: Optional[float]
    
    # Pipeline stages
    scraping_completed: bool
    ai_filtering_completed: bool
    matching_completed: bool
    total_ai_calls: int
    
    # Detailed events
    platform_events: List[Dict[str, Any]]
    ai_events: List[Dict[str, Any]]
    
    # Flexible metadata
    metadata: Dict[str, Any]


# ============================================================================
# Pydantic Models - LEGACY (backward compatible)

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
    platform: Optional[str] = None  # Platform being scraped (Zepto, Blinkit, etc.)
    endpoint: Optional[str] = None  # 'ai-extract', 'smart-search', 'match-products'
    ai_provider: str  # 'groq', 'mistral', 'gemini'
    ai_model: str  # 'mixtral-8x7b', 'gemini-pro', etc.
    fallback_reason: Optional[str] = None
    input_html_size_kb: float = 0.0  # Size of HTML input
    products_found: int = 0  # Products extracted
    products_filtered: int = 0  # Products after filtering
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
        cursor = get_cursor(conn)
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
    print(f"[Analytics] log_bulk_events: Processing {len(logs)} logs")
    with get_db() as conn:
        cursor = get_cursor(conn)
        for i, log in enumerate(logs):
            try:
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
                row_id = cursor.lastrowid
                ids.append(row_id)
                print(f"[Analytics] Inserted log {i+1}/{len(logs)}: ID={row_id}, Platform={log.platform}, Device={log.device_id}")
            except Exception as e:
                print(f"[Analytics] ERROR inserting log {i+1}/{len(logs)}: {str(e)}")
                raise
        print(f"[Analytics] All {len(ids)} logs inserted, awaiting commit")
    print(f"[Analytics] Database committed successfully")
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
        cursor = get_cursor(conn)
        
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
        
        platform_rows = fetchall_as_dicts(cursor)
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
        
        ai_model_usage = {row['model']: row['count'] for row in fetchall_as_dicts(cursor)}
        
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
        
        scrape_source_breakdown = {row['scrape_source']: row['count'] for row in fetchall_as_dicts(cursor)}
        
        # Calculate overall success rate
        cursor.execute("""
            SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) * 100 as overall_success_rate
            FROM scrape_logs
            WHERE device_id = ?
              AND date(created_at) >= date(?)
              AND date(created_at) <= date(?)
        """, (device_id, start_date, end_date))
        
        overall_success_rate = fetchone_as_dict(cursor)['overall_success_rate'] or 0.0
        
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
        cursor = get_cursor(conn)
        
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
        
        cursor.execute(query, tuple(params))
        
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
            for row in fetchall_as_dicts(cursor)
        ]


def get_all_devices() -> List[Dict[str, Any]]:
    """Get list of all devices with their last activity."""
    with get_db() as conn:
        cursor = get_cursor(conn)
        
        # Get devices from search_sessions (primary source)
        cursor.execute("""
            SELECT 
                device_id,
                COUNT(*) as total_sessions,
                MAX(started_at) as last_activity,
                COUNT(DISTINCT date(started_at)) as active_days,
                SUM(total_products_found) as total_products,
                SUM(relevant_products) as total_relevant
            FROM search_sessions
            GROUP BY device_id
            ORDER BY last_activity DESC
        """)
        
        devices = {}
        for row in fetchall_as_dicts(cursor):
            devices[row['device_id']] = {
                "device_id": row['device_id'],
                "total_sessions": row['total_sessions'] or 0,
                "total_logs": row['total_sessions'] or 0,  # Alias for compatibility
                "last_activity": row['last_activity'],
                "active_days": row['active_days'] or 0,
                "total_products": row['total_products'] or 0,
                "total_relevant": row['total_relevant'] or 0
            }
        
        # Also get devices from scrape_logs (legacy)
        cursor.execute("""
            SELECT 
                device_id,
                COUNT(*) as total_logs,
                MAX(created_at) as last_activity,
                COUNT(DISTINCT date(created_at)) as active_days
            FROM scrape_logs
            GROUP BY device_id
        """)
        
        for row in fetchall_as_dicts(cursor):
            device_id = row['device_id']
            if device_id not in devices:
                devices[device_id] = {
                    "device_id": device_id,
                    "total_sessions": 0,
                    "total_logs": row['total_logs'] or 0,
                    "last_activity": row['last_activity'],
                    "active_days": row['active_days'] or 0,
                    "total_products": 0,
                    "total_relevant": 0
                }
            else:
                # Merge - take more recent activity
                existing = devices[device_id]
                existing['total_logs'] += (row['total_logs'] or 0)
                if row['last_activity'] and (not existing['last_activity'] or row['last_activity'] > existing['last_activity']):
                    existing['last_activity'] = row['last_activity']
        
        # Return as sorted list
        return sorted(devices.values(), key=lambda x: x.get('last_activity') or '', reverse=True)


def log_ai_processing(log: AIProcessingLogRequest) -> int:
    """Log AI processing event from backend."""
    with get_db() as conn:
        cursor = get_cursor(conn)
        cursor.execute("""
            INSERT INTO ai_processing_logs (
                device_id, search_query, platform, endpoint, ai_provider, ai_model,
                input_html_size_kb, products_found, products_filtered,
                latency_ms, fallback_reason, success, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log.device_id,
            log.search_query,
            log.platform,
            log.endpoint,
            log.ai_provider,
            log.ai_model,
            log.input_html_size_kb,
            log.products_found,
            log.products_filtered,
            log.latency_ms,
            log.fallback_reason,
            log.success,
            log.error_message
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
        cursor = get_cursor(conn)
        
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
        
        totals = fetchone_as_dict(cursor)
        
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
            for row in fetchall_as_dicts(cursor)
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
            for row in fetchall_as_dicts(cursor)
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


# ============================================================================
# AI Model Accuracy Tracking
# ============================================================================

class AIAccuracyLogRequest(BaseModel):
    """Request to log AI accuracy metrics for model comparison."""
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    search_query: str
    query_category: Optional[str] = None  # 'fruit', 'dairy', etc.
    ai_provider: str
    ai_model: str
    total_input_products: int = 0
    products_kept: int = 0
    products_filtered: int = 0
    high_relevance_count: int = 0  # score >= 80
    medium_relevance_count: int = 0  # 60-79
    low_relevance_count: int = 0  # < 60
    best_deal_relevance_score: Optional[int] = None
    best_deal_reason: Optional[str] = None  # 'exact_match', 'close_match', 'lowest_price'
    latency_ms: int = 0
    success: bool = True


def log_ai_accuracy(log: AIAccuracyLogRequest) -> int:
    """Log AI accuracy metrics for model comparison."""
    with get_db() as conn:
        cursor = get_cursor(conn)
        cursor.execute("""
            INSERT INTO ai_accuracy_logs (
                session_id, device_id, search_query, query_category,
                ai_provider, ai_model,
                total_input_products, products_kept, products_filtered,
                high_relevance_count, medium_relevance_count, low_relevance_count,
                best_deal_relevance_score, best_deal_reason,
                latency_ms, success
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log.session_id,
            log.device_id,
            log.search_query,
            log.query_category,
            log.ai_provider,
            log.ai_model,
            log.total_input_products,
            log.products_kept,
            log.products_filtered,
            log.high_relevance_count,
            log.medium_relevance_count,
            log.low_relevance_count,
            log.best_deal_relevance_score,
            log.best_deal_reason,
            log.latency_ms,
            log.success
        ))
        return cursor.lastrowid


def get_ai_model_accuracy_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Get AI model accuracy comparison stats - which models filter better."""
    if not start_date:
        start_date = (date.today() - timedelta(days=7)).isoformat()
    if not end_date:
        end_date = date.today().isoformat()
    
    with get_db() as conn:
        cursor = get_cursor(conn)
        
        # Get per-model accuracy stats
        cursor.execute("""
            SELECT 
                ai_provider,
                ai_model,
                COUNT(*) as total_calls,
                AVG(latency_ms) as avg_latency_ms,
                SUM(total_input_products) as total_input,
                SUM(products_kept) as total_kept,
                SUM(products_filtered) as total_filtered,
                SUM(high_relevance_count) as high_relevance_total,
                SUM(medium_relevance_count) as medium_relevance_total,
                SUM(low_relevance_count) as low_relevance_total,
                AVG(best_deal_relevance_score) as avg_best_deal_score,
                SUM(CASE WHEN best_deal_reason = 'exact_match' THEN 1 ELSE 0 END) as exact_match_best_deals,
                SUM(CASE WHEN best_deal_reason = 'close_match' THEN 1 ELSE 0 END) as close_match_best_deals,
                SUM(CASE WHEN best_deal_reason = 'lowest_price' THEN 1 ELSE 0 END) as lowest_price_best_deals,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful_calls
            FROM ai_accuracy_logs
            WHERE date(created_at) BETWEEN date(?) AND date(?)
            GROUP BY ai_provider, ai_model
            ORDER BY total_calls DESC
        """, (start_date, end_date))
        
        model_stats = []
        for row in fetchall_as_dicts(cursor):
            total_kept = row['total_kept'] or 0
            high_rel = row['high_relevance_total'] or 0
            total_calls = row['total_calls'] or 0
            
            # Calculate accuracy score (higher = better filtering)
            # Accuracy = (high_relevance / total_kept) * 100
            accuracy_score = (high_rel / total_kept * 100) if total_kept > 0 else 0
            
            # Best deal quality = % of best deals that are exact matches
            exact_matches = row['exact_match_best_deals'] or 0
            close_matches = row['close_match_best_deals'] or 0
            best_deal_quality = ((exact_matches + close_matches * 0.5) / total_calls * 100) if total_calls > 0 else 0
            
            model_stats.append({
                "provider": row['ai_provider'],
                "model": row['ai_model'],
                "total_calls": total_calls,
                "avg_latency_ms": round(row['avg_latency_ms'] or 0, 1),
                "total_input_products": row['total_input'] or 0,
                "products_kept": total_kept,
                "products_filtered": row['total_filtered'] or 0,
                "high_relevance_products": high_rel,
                "medium_relevance_products": row['medium_relevance_total'] or 0,
                "low_relevance_products": row['low_relevance_total'] or 0,
                "accuracy_score": round(accuracy_score, 1),  # % of kept products that are high relevance
                "best_deal_quality": round(best_deal_quality, 1),  # % of best deals that are exact/close match
                "avg_best_deal_score": round(row['avg_best_deal_score'] or 0, 1),
                "exact_match_best_deals": exact_matches,
                "close_match_best_deals": close_matches,
                "lowest_price_best_deals": row['lowest_price_best_deals'] or 0,
                "success_rate": round((row['successful_calls'] / total_calls * 100) if total_calls > 0 else 0, 1)
            })
        
        # Sort by accuracy score
        model_stats.sort(key=lambda x: x['accuracy_score'], reverse=True)
        
        return {
            "date_range": {"start": start_date, "end": end_date},
            "model_comparison": model_stats,
            "best_accuracy_model": model_stats[0] if model_stats else None,
            "best_latency_model": min(model_stats, key=lambda x: x['avg_latency_ms']) if model_stats else None
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
        cursor = get_cursor(conn)
        
        # Get total hits per provider (all time)
        cursor.execute("""
            SELECT 
                ai_provider,
                ai_model,
                COUNT(*) as total_requests,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as failed,
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
            for row in fetchall_as_dicts(cursor)
        ]
        
        # Get today's hits per provider (for daily quota tracking)
        today = date.today().isoformat()
        cursor.execute("""
            SELECT 
                ai_provider,
                COUNT(*) as today_requests,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as today_successful
            FROM ai_processing_logs
            WHERE date(created_at) = date(?)
            GROUP BY ai_provider
        """, (today,))
        
        today_stats = {
            row['ai_provider']: {
                "today_requests": row['today_requests'],
                "today_successful": row['today_successful']
            }
            for row in fetchall_as_dicts(cursor)
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
            for row in fetchall_as_dicts(cursor)
        }
        
        # Get total across all providers
        cursor.execute("""
            SELECT 
                COUNT(*) as total_all_time,
                SUM(CASE WHEN date(created_at) = date(?) THEN 1 ELSE 0 END) as total_today
            FROM ai_processing_logs
        """, (today,))
        
        totals = fetchone_as_dict(cursor)
        
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


def get_app_wide_stats(start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
    """
    Get holistic app-wide statistics across ALL devices.
    
    This is the "bird's eye view" of the entire system:
    - Total searches, products, AI calls across all devices
    - Platform-wise failure rates
    - Best deal success rates
    - AI relevance extraction accuracy
    - Scrape source distribution (device vs AI fallback vs playwright)
    """
    if not start_date:
        start_date = (date.today() - timedelta(days=7)).isoformat()
    if not end_date:
        end_date = date.today().isoformat()
    
    with get_db() as conn:
        cursor = get_cursor(conn)
        
        # Overall scrape stats across ALL devices
        cursor.execute("""
            SELECT 
                COUNT(*) as total_scrapes,
                COUNT(DISTINCT device_id) as unique_devices,
                COUNT(DISTINCT search_query) as unique_queries,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful_scrapes,
                SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as failed_scrapes,
                SUM(products_scraped) as total_products,
                SUM(relevant_products) as total_relevant,
                AVG(latency_ms) as avg_latency_ms
            FROM scrape_logs
            WHERE date(created_at) >= date(?) AND date(created_at) <= date(?)
        """, (start_date, end_date))
        
        scrape_totals = fetchone_as_dict(cursor)
        
        # Scrape source breakdown (device vs AI fallback vs playwright)
        cursor.execute("""
            SELECT 
                scrape_source,
                COUNT(*) as count,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful,
                AVG(products_scraped) as avg_products
            FROM scrape_logs
            WHERE date(created_at) >= date(?) AND date(created_at) <= date(?)
            GROUP BY scrape_source
        """, (start_date, end_date))
        
        scrape_source_stats = [
            {
                "source": row['scrape_source'],
                "count": row['count'],
                "successful": row['successful'],
                "success_rate": round((row['successful'] / row['count'] * 100) if row['count'] > 0 else 0, 1),
                "avg_products": round(row['avg_products'] or 0, 1)
            }
            for row in fetchall_as_dicts(cursor)
        ]
        
        # Platform failure rates
        cursor.execute("""
            SELECT 
                platform,
                COUNT(*) as total,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as failed,
                SUM(products_scraped) as products,
                SUM(relevant_products) as relevant,
                AVG(latency_ms) as avg_latency
            FROM scrape_logs
            WHERE date(created_at) >= date(?) AND date(created_at) <= date(?)
            GROUP BY platform
            ORDER BY total DESC
        """, (start_date, end_date))
        
        platform_stats = [
            {
                "platform": row['platform'],
                "total": row['total'],
                "successful": row['successful'],
                "failed": row['failed'],
                "failure_rate": round((row['failed'] / row['total'] * 100) if row['total'] > 0 else 0, 1),
                "products": row['products'] or 0,
                "relevant": row['relevant'] or 0,
                "relevance_rate": round((row['relevant'] / row['products'] * 100) if row['products'] > 0 else 0, 1),
                "avg_latency_ms": round(row['avg_latency'] or 0, 0)
            }
            for row in fetchall_as_dicts(cursor)
        ]
        
        # AI processing stats (extraction accuracy)
        cursor.execute("""
            SELECT 
                COUNT(*) as total_ai_calls,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful_ai,
                SUM(products_found) as ai_products_found,
                SUM(products_filtered) as ai_products_filtered,
                AVG(latency_ms) as avg_ai_latency
            FROM ai_processing_logs
            WHERE date(created_at) >= date(?) AND date(created_at) <= date(?)
        """, (start_date, end_date))
        
        ai_totals = fetchone_as_dict(cursor)
        
        # AI provider breakdown
        cursor.execute("""
            SELECT 
                ai_provider,
                COUNT(*) as calls,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN fallback_reason IS NOT NULL THEN 1 ELSE 0 END) as fallback_calls,
                AVG(latency_ms) as avg_latency
            FROM ai_processing_logs
            WHERE date(created_at) >= date(?) AND date(created_at) <= date(?)
            GROUP BY ai_provider
            ORDER BY calls DESC
        """, (start_date, end_date))
        
        ai_provider_stats = [
            {
                "provider": row['ai_provider'],
                "calls": row['calls'],
                "successful": row['successful'],
                "success_rate": round((row['successful'] / row['calls'] * 100) if row['calls'] > 0 else 0, 1),
                "fallback_calls": row['fallback_calls'],
                "avg_latency_ms": round(row['avg_latency'] or 0, 0)
            }
            for row in fetchall_as_dicts(cursor)
        ]
        
        # Session stats (best deal success)
        cursor.execute("""
            SELECT 
                COUNT(*) as total_sessions,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN best_deal_product IS NOT NULL THEN 1 ELSE 0 END) as with_best_deal,
                AVG(total_latency_ms) as avg_session_latency,
                AVG(total_products_found) as avg_products_per_session,
                AVG(relevant_products) as avg_relevant_per_session
            FROM search_sessions
            WHERE date(started_at) >= date(?) AND date(started_at) <= date(?)
        """, (start_date, end_date))
        
        session_stats_row = fetchone_as_dict(cursor)
        completed_count = session_stats_row['completed'] or 0
        with_best_deal_count = session_stats_row['with_best_deal'] or 0
        session_stats = {
            "total_sessions": session_stats_row['total_sessions'] or 0,
            "completed": completed_count,
            "failed": session_stats_row['failed'] or 0,
            "with_best_deal": with_best_deal_count,
            "best_deal_rate": round(
                (with_best_deal_count / completed_count * 100) 
                if completed_count > 0 else 0, 1
            ),
            "avg_latency_ms": round(session_stats_row['avg_session_latency'] or 0, 0),
            "avg_products_per_session": round(session_stats_row['avg_products_per_session'] or 0, 1),
            "avg_relevant_per_session": round(session_stats_row['avg_relevant_per_session'] or 0, 1)
        }
        
        # Calculate AI relevance extraction rate
        total_scraped = scrape_totals['total_products'] or 0
        total_relevant = scrape_totals['total_relevant'] or 0
        relevance_rate = round((total_relevant / total_scraped * 100) if total_scraped > 0 else 0, 1)
        
        return {
            "date_range": {"start": start_date, "end": end_date},
            "overview": {
                "total_scrapes": scrape_totals['total_scrapes'] or 0,
                "unique_devices": scrape_totals['unique_devices'] or 0,
                "unique_queries": scrape_totals['unique_queries'] or 0,
                "total_products": total_scraped,
                "total_relevant": total_relevant,
                "relevance_rate": relevance_rate,
                "success_rate": round(
                    (scrape_totals['successful_scrapes'] / scrape_totals['total_scrapes'] * 100) 
                    if scrape_totals['total_scrapes'] > 0 else 0, 1
                ),
                "avg_latency_ms": round(scrape_totals['avg_latency_ms'] or 0, 0)
            },
            "scrape_sources": scrape_source_stats,
            "platforms": platform_stats,
            "ai_processing": {
                "total_calls": ai_totals['total_ai_calls'] or 0,
                "successful": ai_totals['successful_ai'] or 0,
                "products_found": ai_totals['ai_products_found'] or 0,
                "products_filtered": ai_totals['ai_products_filtered'] or 0,
                "avg_latency_ms": round(ai_totals['avg_ai_latency'] or 0, 0)
            },
            "ai_providers": ai_provider_stats,
            "sessions": session_stats
        }


def get_recent_sessions(
    device_id: Optional[str] = None,
    limit: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict]:
    """
    Get recent search sessions for display in dashboard.
    
    If device_id is None, returns sessions across ALL devices.
    Each session includes summary info and a link to drill-down view.
    """
    if not start_date:
        start_date = (date.today() - timedelta(days=7)).isoformat()
    if not end_date:
        end_date = date.today().isoformat()
    
    with get_db() as conn:
        cursor = get_cursor(conn)
        
        if device_id:
            cursor.execute("""
                SELECT 
                    session_id, device_id, search_query, pincode,
                    started_at, completed_at, total_latency_ms,
                    total_platforms, successful_platforms,
                    total_products_found, relevant_products,
                    best_deal_product, best_deal_price, best_deal_platform,
                    status
                FROM search_sessions
                WHERE device_id = ?
                  AND date(started_at) >= date(?)
                  AND date(started_at) <= date(?)
                ORDER BY started_at DESC
                LIMIT ?
            """, (device_id, start_date, end_date, limit))
        else:
            cursor.execute("""
                SELECT 
                    session_id, device_id, search_query, pincode,
                    started_at, completed_at, total_latency_ms,
                    total_platforms, successful_platforms,
                    total_products_found, relevant_products,
                    best_deal_product, best_deal_price, best_deal_platform,
                    status
                FROM search_sessions
                WHERE date(started_at) >= date(?)
                  AND date(started_at) <= date(?)
                ORDER BY started_at DESC
                LIMIT ?
            """, (start_date, end_date, limit))
        
        sessions = []
        for row in fetchall_as_dicts(cursor):
            sessions.append({
                "session_id": row['session_id'],
                "device_id": row['device_id'],
                "device_id_short": row['device_id'][:8] + "..." if row['device_id'] else "unknown",
                "search_query": row['search_query'],
                "pincode": row['pincode'],
                "started_at": row['started_at'],
                "completed_at": row['completed_at'],
                "total_latency_ms": row['total_latency_ms'],
                "total_platforms": row['total_platforms'],
                "successful_platforms": row['successful_platforms'],
                "total_products": row['total_products_found'],
                "relevant_products": row['relevant_products'],
                "best_deal": {
                    "product": row['best_deal_product'],
                    "price": row['best_deal_price'],
                    "platform": row['best_deal_platform']
                } if row['best_deal_product'] else None,
                "status": row['status'],
                "pipeline_url": f"/session?id={row['session_id']}"
            })
        
        return sessions


# ============================================================================
# Session-Based Analytics Functions
# ============================================================================

import json

def create_session(request: CreateSessionRequest) -> str:
    """Create a new search session."""
    with get_db() as conn:
        cursor = get_cursor(conn)
        metadata_json = json.dumps(request.metadata or {})
        cursor.execute("""
            INSERT INTO search_sessions (
                session_id, device_id, search_query, pincode, total_platforms, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            request.session_id,
            request.device_id,
            request.search_query,
            request.pincode,
            request.total_platforms,
            metadata_json
        ))
        return request.session_id


def update_session(request: UpdateSessionRequest) -> bool:
    """Update session after search completes."""
    with get_db() as conn:
        cursor = get_cursor(conn)
        
        # Build dynamic update query
        updates = []
        values = []
        
        if request.completed_at:
            updates.append("completed_at = ?")
            values.append(request.completed_at)
        if request.total_latency_ms is not None:
            updates.append("total_latency_ms = ?")
            values.append(request.total_latency_ms)
        if request.successful_platforms is not None:
            updates.append("successful_platforms = ?")
            values.append(request.successful_platforms)
        if request.failed_platforms is not None:
            updates.append("failed_platforms = ?")
            values.append(request.failed_platforms)
        if request.total_products_found is not None:
            updates.append("total_products_found = ?")
            values.append(request.total_products_found)
        if request.relevant_products is not None:
            updates.append("relevant_products = ?")
            values.append(request.relevant_products)
        if request.filtered_out is not None:
            updates.append("filtered_out = ?")
            values.append(request.filtered_out)
        if request.best_deal_platform:
            updates.append("best_deal_platform = ?")
            values.append(request.best_deal_platform)
        if request.best_deal_product:
            updates.append("best_deal_product = ?")
            values.append(request.best_deal_product)
        if request.best_deal_price is not None:
            updates.append("best_deal_price = ?")
            values.append(request.best_deal_price)
        if request.scraping_completed is not None:
            updates.append("scraping_completed = ?")
            values.append(request.scraping_completed)
        if request.ai_filtering_completed is not None:
            updates.append("ai_filtering_completed = ?")
            values.append(request.ai_filtering_completed)
        if request.matching_completed is not None:
            updates.append("matching_completed = ?")
            values.append(request.matching_completed)
        if request.total_ai_calls is not None:
            updates.append("total_ai_calls = ?")
            values.append(request.total_ai_calls)
        if request.status:
            updates.append("status = ?")
            values.append(request.status)
        if request.error_message:
            updates.append("error_message = ?")
            values.append(request.error_message)
        if request.metadata:
            # Merge with existing metadata
            cursor.execute("SELECT metadata FROM search_sessions WHERE session_id = ?", (request.session_id,))
            row = fetchone_as_dict(cursor)
            existing = json.loads(row['metadata']) if row and row['metadata'] else {}
            existing.update(request.metadata)
            updates.append("metadata = ?")
            values.append(json.dumps(existing))
        
        if not updates:
            return False
        
        values.append(request.session_id)
        query = f"UPDATE search_sessions SET {', '.join(updates)} WHERE session_id = ?"
        cursor.execute(query, tuple(values))  # Convert to tuple for libsql compatibility
        return cursor.rowcount > 0


def log_platform_scrape_event(request: PlatformScrapeEventRequest) -> int:
    """Log a platform scrape event within a session."""
    with get_db() as conn:
        cursor = get_cursor(conn)
        metadata_json = json.dumps(request.metadata or {})
        cursor.execute("""
            INSERT INTO platform_scrape_events (
                session_id, device_id, platform, scrape_source, scrape_tier,
                html_size_kb, html_truncated, products_found, relevant_products,
                latency_ms, success, error_message, retry_count, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request.session_id,
            request.device_id,
            request.platform,
            request.scrape_source,
            request.scrape_tier,
            request.html_size_kb,
            request.html_truncated,
            request.products_found,
            request.relevant_products,
            request.latency_ms,
            request.success,
            request.error_message,
            request.retry_count,
            metadata_json
        ))
        return cursor.lastrowid


def log_ai_processing_event(request: AIProcessingEventRequest) -> int:
    """Log an AI processing event within a session."""
    print(f"[Analytics] log_ai_processing_event called: session_id={request.session_id}, device_id={request.device_id}, endpoint={request.endpoint}, provider={request.ai_provider}")
    try:
        with get_db() as conn:
            cursor = get_cursor(conn)
            metadata_json = json.dumps(request.metadata or {})
            cursor.execute("""
                INSERT INTO ai_processing_events (
                    session_id, device_id, endpoint, platform, ai_provider, ai_model,
                    is_fallback, fallback_reason, input_size_kb, input_tokens, output_tokens,
                    products_input, products_output, latency_ms, success, error_message,
                    http_status, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request.session_id,
                request.device_id,
                request.endpoint,
                request.platform,
                request.ai_provider,
                request.ai_model,
                request.is_fallback,
                request.fallback_reason,
                request.input_size_kb,
                request.input_tokens,
                request.output_tokens,
                request.products_input,
                request.products_output,
                request.latency_ms,
                request.success,
                request.error_message,
                request.http_status,
                metadata_json
            ))
            event_id = cursor.lastrowid
            print(f"[Analytics] AI event logged successfully with id={event_id}")
            return event_id
    except Exception as e:
        print(f"[Analytics] ERROR logging AI event: {e}")
        import traceback
        traceback.print_exc()
        return -1


def get_session_detail(session_id: str) -> Optional[Dict[str, Any]]:
    """Get full session detail with all events."""
    with get_db() as conn:
        cursor = get_cursor(conn)
        
        # Get session
        cursor.execute("SELECT * FROM search_sessions WHERE session_id = ?", (session_id,))
        session = fetchone_as_dict(cursor)
        if not session:
            return None
        
        # Get platform events
        cursor.execute("""
            SELECT * FROM platform_scrape_events 
            WHERE session_id = ? 
            ORDER BY created_at
        """, (session_id,))
        platform_events = [dict(row) for row in fetchall_as_dicts(cursor)]
        
        # Get AI events (match by session_id OR device_id for events logged during session time window)
        device_id = session.get('device_id')
        started_at = session.get('started_at')
        completed_at = session.get('completed_at')
        
        cursor.execute("""
            SELECT * FROM ai_processing_events 
            WHERE session_id = ? 
               OR (device_id = ? AND created_at >= ? AND (created_at <= ? OR ? IS NULL))
            ORDER BY created_at
        """, (session_id, device_id, started_at, completed_at, completed_at))
        ai_events = [dict(row) for row in fetchall_as_dicts(cursor)]
        
        # Parse metadata
        session_dict = dict(session)
        session_dict['metadata'] = json.loads(session_dict.get('metadata') or '{}')
        for event in platform_events:
            event['metadata'] = json.loads(event.get('metadata') or '{}')
        for event in ai_events:
            event['metadata'] = json.loads(event.get('metadata') or '{}')
        
        return {
            **session_dict,
            'platform_events': platform_events,
            'ai_events': ai_events
        }


def get_device_sessions(
    device_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Get recent sessions for a device."""
    if not start_date:
        start_date = date.today().isoformat()
    if not end_date:
        end_date = date.today().isoformat()
    
    with get_db() as conn:
        cursor = get_cursor(conn)
        cursor.execute("""
            SELECT 
                session_id, search_query, pincode, started_at, completed_at,
                total_latency_ms, total_platforms, successful_platforms, failed_platforms,
                total_products_found, relevant_products, best_deal_platform,
                best_deal_product, best_deal_price, total_ai_calls, status
            FROM search_sessions
            WHERE device_id = ?
              AND date(started_at) BETWEEN date(?) AND date(?)
            ORDER BY started_at DESC
            LIMIT ?
        """, (device_id, start_date, end_date, limit))
        
        return [dict(row) for row in fetchall_as_dicts(cursor)]


def get_session_pipeline_visualization(session_id: str) -> Dict[str, Any]:
    """Get session data formatted for pipeline visualization."""
    detail = get_session_detail(session_id)
    if not detail:
        return {"error": "Session not found"}
    
    # Build pipeline stages
    pipeline = {
        "session": {
            "session_id": detail['session_id'],
            "query": detail['search_query'],
            "device_id": detail['device_id'],
            "pincode": detail.get('pincode'),
            "started_at": detail['started_at'],
            "completed_at": detail.get('completed_at'),
            "total_latency_ms": detail.get('total_latency_ms', 0),
            "status": detail.get('status', 'unknown')
        },
        "stage_1_scraping": {
            "name": "Platform Scraping",
            "total_platforms": detail.get('total_platforms', 0),
            "successful": detail.get('successful_platforms', 0),
            "failed": detail.get('failed_platforms', 0),
            "platforms": []
        },
        "stage_2_ai_extraction": {
            "name": "AI Product Extraction",
            "calls": []
        },
        "stage_3_filtering": {
            "name": "AI Filtering",
            "calls": []
        },
        "stage_4_matching": {
            "name": "Product Matching",
            "calls": []
        },
        "summary": {
            "total_products_found": detail.get('total_products_found', 0),
            "relevant_products": detail.get('relevant_products', 0),
            "filtered_out": detail.get('filtered_out', 0),
            "total_ai_calls": detail.get('total_ai_calls', 0),
            "best_deal": {
                "platform": detail.get('best_deal_platform'),
                "product": detail.get('best_deal_product'),
                "price": detail.get('best_deal_price')
            } if detail.get('best_deal_platform') else None
        }
    }
    
    # Process platform events
    for event in detail.get('platform_events', []):
        pipeline['stage_1_scraping']['platforms'].append({
            "platform": event['platform'],
            "source": event['scrape_source'],
            "tier": event.get('scrape_tier', 1),
            "html_size_kb": event.get('html_size_kb', 0),
            "products_found": event.get('products_found', 0),
            "relevant_products": event.get('relevant_products', 0),
            "latency_ms": event.get('latency_ms', 0),
            "success": event.get('success', False),
            "error": event.get('error_message'),
            "retry_count": event.get('retry_count', 0)
        })
    
    # Process AI events by endpoint
    for event in detail.get('ai_events', []):
        endpoint = event.get('endpoint', 'unknown')
        ai_call = {
            "platform": event.get('platform'),
            "provider": event['ai_provider'],
            "model": event['ai_model'],
            "is_fallback": event.get('is_fallback', False),
            "fallback_reason": event.get('fallback_reason'),
            "input_size_kb": event.get('input_size_kb', 0),
            "products_input": event.get('products_input', 0),
            "products_output": event.get('products_output', 0),
            "latency_ms": event.get('latency_ms', 0),
            "success": event.get('success', False),
            "error": event.get('error_message')
        }
        
        if 'extract' in endpoint.lower():
            pipeline['stage_2_ai_extraction']['calls'].append(ai_call)
        elif 'filter' in endpoint.lower() or 'smart-search' in endpoint.lower():
            pipeline['stage_3_filtering']['calls'].append(ai_call)
        elif 'match' in endpoint.lower():
            pipeline['stage_4_matching']['calls'].append(ai_call)
    
    return pipeline


# Initialize database on module import
init_database()
