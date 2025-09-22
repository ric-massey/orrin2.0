# goals/__init__.py
# Orrin Goals package initializer

# Re-export common submodules for convenience (optional).
# Import failures here shouldn’t break the package, so keep it light.
__all__ = [
    "cli", "api", "daemon", "events", "health", "locks", "metrics",
    "model", "policy", "registry", "runner", "schema", "snapshot",
    "store", "triggers", "utils", "wal",
]
