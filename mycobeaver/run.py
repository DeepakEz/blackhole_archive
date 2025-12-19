#!/usr/bin/env python
"""
Simple runner script for MycoBeaver.
Run this from inside the mycobeaver directory:
    python run.py --mode train --scenario small
"""
import sys
import os

# Add parent directory to path so imports work
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Now import and run
from mycobeaver.main import main

if __name__ == "__main__":
    main()
