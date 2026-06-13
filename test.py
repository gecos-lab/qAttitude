#!/usr/bin/env python3
"""
Simple test runner script for qAttitude
Works on Windows, macOS, and Linux
"""

import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Add repo root to Python path
    sys.path.insert(0, str(script_dir))
    
    print("=" * 70)
    print("qAttitude Test Runner")
    print("=" * 70)
    print(f"Repository root: {script_dir}")
    print(f"Python version: {sys.version}")
    print()
    
    # Check if requirements are installed
    print("Checking dependencies...")
    required_packages = [
        'pandas', 'mplstereonet', 'sklearn', 'kmedoids', 'sphstat'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing_packages.append(package)
    
    if missing_packages:
        print()
        print("ERROR: Missing dependencies!")
        print(f"Run: pip install -r requirements.txt")
        print()
        return 1
    
    print()
    print("=" * 70)
    print("Running tests...")
    print("=" * 70)
    print()
    
    # Run unittest discovery
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'unittest', 'discover', 'tests', '-v'],
            cwd=script_dir,
            capture_output=False
        )
        return result.returncode
    except Exception as e:
        print(f"ERROR: Failed to run tests: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
