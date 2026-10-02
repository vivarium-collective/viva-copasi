"""Visualization Step subclasses for viva-copasi.

Visualizations follow the pbg-superpowers new-style contract: each
subclass implements ``accumulate(state)`` to buffer per-step numeric
data (like an Emitter) and ``render()`` to build the Plotly figure once
at end-of-run, returning the HTML string.  The base orchestrator owns
the per-tick path (it calls ``accumulate`` each step and ``render``
once), so subclasses do NOT override ``update()``.  The composite spec
wires the input ports to store paths.

See viva_superpowers.visualization for the base-class contract.
"""
from __future__ import annotations

from viva_superpowers.visualization import Visualization


class SpeciesConcentrationsPlot(Visualization):
    """Time-series HTML plot of CopasiUTCProcess's species concentrations.

    Buffers the ``species_concentrations`` map (and optionally ``time``)
    at each step into per-species trajectories, then renders a single
    Plotly HTML figure at end-of-run.  Downstream consumers (dashboards,
    notebook viewers) read the rendered 'html' from the wired store.

    Input port shape:

    - ``species_concentrations``: map[float]  (SBML IDs → concentration)
    - ``time``:  float (optional; falls back to step counter)
    """

    config_schema = {
        'title': {'_type': 'string', '_default': 'COPASI species concentrations'},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times: list[float] = []
        # species_id -> list of values, lazily populated as new species appear
        self.history: dict[str, list[float]] = {}

    def inputs(self):
        return {
            'species_concentrations': 'map[float]',
            'time': 'float',
        }

    def accumulate(self, state):
        t = state.get('time')
        if t is None:
            t = float(len(self.times))
        self.times.append(float(t))

        species = state.get('species_concentrations') or {}
        # Extend any newly-seen species with zeros so series stay aligned.
        n = len(self.times)
        for sid in species:
            if sid not in self.history:
                self.history[sid] = [0.0] * (n - 1)
        # Append the current sample for every known species.
        for sid in list(self.history.keys()):
            v = species.get(sid)
            self.history[sid].append(float(v) if v is not None else 0.0)

    def render(self):
        title = (self.config or {}).get('title', 'COPASI species concentrations')
        traces = []
        for sid, ys in self.history.items():
            traces.append(
                '{"x":' + repr(self.times) + ',"y":' + repr(ys) +
                ',"type":"scatter","mode":"lines","name":"' + sid + '"}'
            )
        html = (
            f'<div id="scplot" style="height:380px"></div>'
            f'<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            f'<script>Plotly.newPlot("scplot",[{",".join(traces)}],'
            f'{{title:"{title}",margin:{{l:55,r:15,t:35,b:40}},'
            f'xaxis:{{title:"time"}},'
            f'yaxis:{{title:"concentration"}},'
            f'legend:{{orientation:"h",y:-0.2}}}},'
            f'{{responsive:true,displayModeBar:false}});</script>'
        )
        return html
