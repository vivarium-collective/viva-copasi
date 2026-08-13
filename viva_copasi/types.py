"""Custom bigraph-schema types for viva-copasi.

Currently uses built-in types (map[float], list, any, etc.) only.
This hook exists for future custom type registrations — e.g.
species-aware map types or unit-bearing concentrations.
"""


def register_copasi_types(core):
    """Register custom types used by COPASI processes. No-op for now."""
    return core
