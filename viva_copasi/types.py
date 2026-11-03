"""Custom bigraph-schema types for viva-copasi.

``standard_types`` is the table compose-api's runner registers (``COMPOSE_RUNNER_TYPES`` names
``viva_copasi:standard_types``). Its ``numeric_result`` is the result shape the steps emit (``time``, ``columns``,
``values``), with each field overwritten rather than accumulated, so a result replaces the one before it. The same
shape is pbsim_common's ``numeric_result``, so the two packages' results are interchangeable on the wire.
"""

standard_types = {
    "numeric_result": {
        "time": "overwrite[list[float]]",
        "columns": "overwrite[list[string]]",
        "values": "overwrite[list[list[float]]]",
    },
    "numeric_results": "map[numeric_result]",
}


def register_copasi_types(core):
    """Register ``standard_types`` on ``core``; types already registered are left as they are."""
    for name, schema in standard_types.items():
        if name not in core.registry:
            core.register_type(name, schema)
    return core
