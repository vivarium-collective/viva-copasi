"""build_core(core=None) — the viva-copasi core.

Cross-repo convention: build_core either creates a fresh process-bigraph core
via ``process_bigraph.allocate_core()`` (when ``core is None``) or composes onto
a passed-in ``core``, registers THIS repo's own process/step classes by name so
they are first-class, browsable dashboard Registry entries, and returns ``core``.
Uniform across viva-/pbg- repos.

``register_package_processes`` (from viva_superpowers.core_compose) does the
registration: it filters to real process_bigraph Process/Step subclasses and is
best-effort + idempotent. It scans the *top-level modules of a package* (via the
package's ``__path__``). viva-copasi keeps its processes in the module
``viva_copasi/processes.py`` (CopasiUTCStep, CopasiSteadyStateStep,
CopasiUTCProcess, ParameterEstimationStep) plus the Visualization Step in
``viva_copasi/visualizations.py``; a module has no ``__path__``, so we pass the
package ``"viva_copasi"`` — the helper walks its top-level modules and registers
every Process/Step subclass defined in them.
"""
from __future__ import annotations

from process_bigraph import allocate_core
from viva_superpowers.core_compose import register_package_processes


def build_core(core=None):
    core = core if core is not None else allocate_core()
    # Register the schema types THIS repo's step outputs declare (#25). The
    # steps emit 'numeric_result' (CopasiUTCStep) and 'any' (CopasiSteadyState-
    # Step); neither is in bigraph-schema's BASE_TYPES, so a document using the
    # steps fails to load under build_core() with a confusing
    # "TypeError: 'str' object does not support item assignment" during wiring.
    # register_copasi() registers them as tree aliases; mirror that here so the
    # documented build_core() core resolves the wiring too. Inlined (not
    # imported from viva_copasi.composites) to keep build_core independent of
    # the composites/visualization import chain.
    if 'any' not in core.registry:
        core.register_type('any', {'_inherit': 'tree'})
    if 'numeric_result' not in core.registry:
        core.register_type('numeric_result', {'_inherit': 'tree'})
    # Register THIS repo's own process/step classes as first-class, browsable
    # Registry entries (a composite that instantiates a process directly does
    # NOT auto-register it by name).
    register_package_processes(core, "viva_copasi")
    return core
