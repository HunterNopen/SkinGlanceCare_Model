import threading

class NullObject:
    """A Null Object that safely absorbs interactions."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self

    def __setattr__(self, name, value):
        # Ignore any attribute assignments except internal
        if name in {"_instance"}:
            return super().__setattr__(name, value)
        return self

    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False