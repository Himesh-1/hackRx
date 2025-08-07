
import sqlite3
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CACHE_DB_PATH = "data/cache.db"
CACHE_EXPIRATION_DAYS = 7 # Cache entries expire after 7 days

def init_cache_db():
    """Initializes the SQLite database for caching LLM responses and query rewrites."""
    conn = None
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()
        logger.info(f"Cache database initialized at {CACHE_DB_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Error initializing cache database: {e}")
    finally:
        if conn:
            conn.close()

def get_cache(key: str) -> any:
    """Retrieves a value from the cache if it's not expired."""
    conn = None
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT value, timestamp FROM llm_cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            value, timestamp_str = row
            stored_time = datetime.fromisoformat(timestamp_str)
            if datetime.now() - stored_time < timedelta(days=CACHE_EXPIRATION_DAYS):
                logger.info(f"Cache hit for key: {key}")
                return json.loads(value)
            else:
                logger.info(f"Cache expired for key: {key}")
                delete_cache(key) # Clean up expired entry
        logger.info(f"Cache miss for key: {key}")
        return None
    except sqlite3.Error as e:
        logger.error(f"Error retrieving from cache: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding cached JSON for key {key}: {e}")
        delete_cache(key) # Remove corrupted entry
        return None
    finally:
        if conn:
            conn.close()

def set_cache(key: str, value: any):
    """Stores a key-value pair in the cache."""
    conn = None
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        timestamp_str = datetime.now().isoformat()
        json_value = json.dumps(value)
        cursor.execute("INSERT OR REPLACE INTO llm_cache (key, value, timestamp) VALUES (?, ?, ?)",
                       (key, json_value, timestamp_str))
        conn.commit()
        logger.info(f"Cache set for key: {key}")
    except sqlite3.Error as e:
        logger.error(f"Error setting cache for key {key}: {e}")
    finally:
        if conn:
            conn.close()

def delete_cache(key: str):
    """Deletes a specific key from the cache."""
    conn = None
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
        conn.commit()
        logger.info(f"Cache deleted for key: {key}")
    except sqlite3.Error as e:
        logger.error(f"Error deleting cache for key {key}: {e}")
    finally:
        if conn:
            conn.close()

def clear_all_cache():
    """Clears all entries from the cache."""
    conn = None
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM llm_cache")
        conn.commit()
        logger.info("All cache entries cleared.")
    except sqlite3.Error as e:
        logger.error(f"Error clearing all cache: {e}")
    finally:
        if conn:
            conn.close()

# Initialize the database when the module is imported
init_cache_db()
