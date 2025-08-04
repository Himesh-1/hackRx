
"""
Utilities Module
Contains shared helper functions and constants.
"""

import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a string to be safe for use as a filename.

    Args:
        filename (str): The original filename string.

    Returns:
        str: A sanitized filename string.
    """
    # Remove invalid characters
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
    # Replace spaces with underscores
    s = s.replace(' ', '_')
    # Limit length
    s = s[:200]
    return s
