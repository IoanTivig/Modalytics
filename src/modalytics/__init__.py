"""
Modalytics: open-source modular multi-modal imaging analysis platform.

Copyright (C) 2026 Ioan Tivig.
Licensed under the GNU Lesser General Public License v3.0 or later
(see COPYING.LESSER and COPYING).
"""

import sys

__version__ = "0.1.0"

# Console messages use emoji prefixes (✅ ⚠️ ℹ️ ❌ 🆕). Windows consoles default
# to a legacy code page that cannot encode them, so switch stdout/stderr to UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
