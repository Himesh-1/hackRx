
import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_FILE = Path("data/unanswered_questions.db")

def view_questions(document: str = None, answered_correctly: bool = None):
    """
    Views questions from the database, with optional filtering.

    Args:
        document: Optional. Filter by document name.
        answered_correctly: Optional. Filter by whether the question was answered correctly.
    """
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            query = "SELECT question, document, timestamp, answered_correctly FROM unanswered_questions WHERE 1=1"
            params = []

            if document:
                query += " AND document = ?"
                params.append(document)
            if answered_correctly is not None:
                query += " AND answered_correctly = ?"
                params.append(1 if answered_correctly else 0)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                print("No questions found matching the criteria.")
                return

            print("\n--- Logged Questions ---")
            for row in rows:
                status = "Answered Correctly" if row[3] else "NOT Answered Correctly"
                print(f"Question: {row[0]}")
                print(f"  Document: {row[1]}")
                print(f"  Timestamp: {row[2]}")
                print(f"  Status: {status}")
                print("------------------------")

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")

if __name__ == "__main__":
    print("\nUsage: python view_questions.py [OPTIONS]")
    print("Options:")
    print("  --document <filename>    Filter by document name (e.g., policy.pdf)")
    print("  --answered <true/false>  Filter by answered status (true for answered, false for not)")
    print("\nExample: python view_questions.py --answered false")
    print("Example: python view_questions.py --document policy.pdf --answered true")

    import argparse
    parser = argparse.ArgumentParser(description="View logged questions.")
    parser.add_argument("--document", type=str, help="Filter by document name.")
    parser.add_argument("--answered", type=str, help="Filter by answered status (true/false).")
    args = parser.parse_args()

    answered_status = None
    if args.answered:
        if args.answered.lower() == "true":
            answered_status = True
        elif args.answered.lower() == "false":
            answered_status = False
        else:
            print("Invalid value for --answered. Use 'true' or 'false'.")
            exit()

    view_questions(document=args.document, answered_correctly=answered_status)
