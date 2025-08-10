
import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_FILE = Path("data/unanswered_questions.db")

def initialize_db():
    """
    Initializes the database and creates the unanswered_questions table if it doesn't exist.
    """
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS unanswered_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    document TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    answered_correctly BOOLEAN NOT NULL
                )
            """)
            conn.commit()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")

def log_question(question: str, document: str, answered_correctly: bool):
    """
    Logs a question to the database.

    Args:
        question: The question that was asked.
        document: The document the question was asked about.
        answered_correctly: Whether the question was answered correctly or not.
    """
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO unanswered_questions (question, document, answered_correctly)
                VALUES (?, ?, ?)
            """, (question, document, answered_correctly))
            conn.commit()
        logger.info(f"Logged question: {question}")
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
