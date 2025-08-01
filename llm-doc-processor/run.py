"""
Run script for the LLM Document Processor application.
This script sets up the Python path and launches the Uvicorn server.
"""

import os
import sys
import uvicorn

if __name__ == "__main__":
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    # Add the site-packages directory to the Python path
    site_packages = os.path.join(project_root, '..', 'venv', 'Lib', 'site-packages')
    sys.path.insert(0, site_packages)

    # Run the Uvicorn server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000
    )