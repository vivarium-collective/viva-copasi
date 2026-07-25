"""Visualization Step subclasses for pbg-copasi.

Visualizations follow the pbg-superpowers convention (v0.4+):
each subclass overrides ``update()`` to consume per-step state via wires
(like an Emitter), accumulates history internally, and returns
``{'html': '<rendered figure>'}`` each step.  The composite spec wires
the input ports to store paths.

See pbg_superpowers.visualization for the base-class contract.
"""
from __future__ import annotations

from pbg_superpowers.visualization import Visualization


class SpeciesConcentrationsPlot(Visualization):
    """Time-series HTML plot of CopasiUTCProcess's species concentrations.

    Consumes the ``species_concentrations`` map (and optionally ``time``)
    at each step, accumulates per-species trajectories across calls, and
    emits a Plotly HTML figure on every update.  Downstream consumers
    (dashboards, notebook viewers) read the latest 'html' from the wired
    store.

    Input port shape:

    - ``species_concentrations``: map[float]  (SBML IDs → concentration)
    - ``time``:  float (optional; falls back to step counter × interval)
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

    def update(self, state, interval=1.0):
        t = state.get('time')
        if t is None:
            t = len(self.times) * (interval or 1.0)
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
        return {'html': html}
