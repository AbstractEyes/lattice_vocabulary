# ============================================================================
# Cache Implementation
# ============================================================================
from collections import OrderedDict

import threading


class LRUCache(OrderedDict):
    """Thread-safe LRU cache"""

    def __init__(self, maxsize: int = 128):
        super().__init__()
        self.maxsize = maxsize
        self._lock = threading.RLock()

    def __getitem__(self, key):
        with self._lock:
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value

    def __setitem__(self, key, value):
        with self._lock:
            if key in self:
                self.move_to_end(key)
            super().__setitem__(key, value)
            if len(self) > self.maxsize:
                oldest = next(iter(self))
                del self[oldest]

    def get(self, key, default=None):
        with self._lock:
            if key in self:
                return self[key]
            return default

    def clear(self):
        """Clear the cache"""
        with self._lock:
            super().clear()