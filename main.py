"""
Convenience launcher (e.g. PyCharm "Run main.py"): same as ``python -m modalytics``.

Usage:
    python main.py [--theme light] [files…]
"""

import os
import sys

# Allow running from a fresh checkout without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from modalytics.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
