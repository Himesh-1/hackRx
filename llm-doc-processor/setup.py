
"""
Setup Script
Installs all necessary dependencies from the requirements.txt file.
"""

import sys
import subprocess
import os

def install_dependencies():
    """Installs dependencies from requirements.txt using pip."""
    print("--- Installing required packages ---")
    try:
        python_executable = sys.executable
        requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
        
        if not os.path.exists(requirements_path):
            print(f"ERROR: requirements.txt not found at {requirements_path}")
            return

        print(f"Using Python interpreter: {python_executable}")
        subprocess.check_call([python_executable, "-m", "pip", "install", "-r", requirements_path])
        print("\n--- Dependencies installed successfully! ---")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    install_dependencies()

