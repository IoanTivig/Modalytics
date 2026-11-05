"""The installed program's entry point (``packaging/modalytics.spec``): ``modalytics.__main__.main``."""

import multiprocessing
import sys

from modalytics.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()                   # a library's worker process must not start the app again
    sys.exit(main())
