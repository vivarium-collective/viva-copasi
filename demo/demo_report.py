"""Demo: COPASI multi-configuration simulation report.

Runs three distinct configurations of the bundled Repressilator SBML model
(Elowitz & Leibler 2000, BIOMD0000000012) using three COPASI-backed
process-bigraph classes:

  1. CopasiUTCStep         — one-shot UTC trajectory
  2. CopasiUTCProcess      — incremental UTC (multi-step)
  3. CopasiSteadyStateStep — steady-state solve

Generates a self-contained HTML report with Plotly time-series charts,
bigraph-viz architecture diagrams, and interactive PBG composite-document
trees.
"""

import base64
import json
import os
import subprocess
import tempfile
import time as _time
from importlib.resources import files

from process_bigraph import allocate_core
from viva_copasi.processes import (
    CopasiUTCStep,
    CopasiUTCProcess,
    CopasiSteadyStateStep,
)
from viva_copasi.composites import (
    register_copasi,
    make_copasi_utc_document,
    make_copasi_utc_process_document,
    make_copasi_steady_state_document,
)

# Bundled repressilator model path
_MODEL_PATH = str(
    files('viva_copasi.composites').joinpath('repressilator.xml')
)


# ── Configs ──────────────────────────────────────────────────────────

CONFIGS = [
    {
        'id': 'utc_step',
        'title': 'UTC Step (one-shot)',
        'subtitle': 'Single-call trajectory via CopasiUTCStep',
        'description': (
            'CopasiUTCStep runs a full COPASI uniform time course in one call. '
            'The entire trajectory from t=0 to t=100 is returned as a result dict '
            'with time points and column values. Demonstrates the Step interface '
            'where no loop is needed — one update() returns the full time series.'
        ),
        'class': CopasiUTCStep,
        'proc_cfg': {
            'model_source': _MODEL_PATH,
            'time': 100.0,
            'n_points': 101,
        },
        'color_scheme': 'indigo',
        'mode': 'step',
    },
    {
        'id': 'utc_process',
        'title': 'UTC Process (incremental)',
        'subtitle': 'Stepwise integration via CopasiUTCProcess',
        'description': (
            'CopasiUTCProcess drives the Repressilator forward one interval at a time '
            'using the Process interface. Each update() call advances the model by '
            '5 time units, allowing composition with other processes in a bigraph '
            'composite. Shows the multi-step, interactive simulation pattern.'
        ),
        'class': CopasiUTCProcess,
        'proc_cfg': {
            'model_source': _MODEL_PATH,
            'time': 5.0,
            'intervals': 10,
        },
        'n_steps': 20,   # 20 steps × 5 time units = 100 time units total
        'color_scheme': 'emerald',
        'mode': 'process',
    },
    {
        'id': 'steady_state',
        'title': 'Steady-State Solve',
        'subtitle': 'Equilibrium concentrations via CopasiSteadyStateStep',
        'description': (
            'CopasiSteadyStateStep invokes COPASI\'s steady-state task to find the '
            'equilibrium concentrations and reaction fluxes for the Repressilator '
            'network. Because the repressilator is an oscillator it may not converge '
            'to a stable fixed point, but the solver result is reported as-is, '
            'demonstrating the steady-state interface.'
        ),
        'class': CopasiSteadyStateStep,
        'proc_cfg': {
            'model_source': _MODEL_PATH,
        },
        'color_scheme': 'rose',
        'mode': 'steady_state',
    },
]


# ── Simulation runners ────────────────────────────────────────────────

def run_utc_step(cfg):
    """Run a CopasiUTCStep — returns (species_ids, columns, times, values, runtime)."""
    core = register_copasi()
    t0 = _time.perf_counter()

    proc = cfg['class'](config=cfg['proc_cfg'], core=core)
    state0 = proc.initial_state()
    result = proc.update({'species_concentrations': {}, 'species_counts': {}})
    runtime = _time.perf_counter() - t0

    r = result['result']
    times = r['time']
    columns = r['columns']
    values = r['values']  # list of rows

    species_ids = proc.species_ids
    return species_ids, columns, times, values, runtime


def run_utc_process(cfg):
    """Run a CopasiUTCProcess for n_steps — returns (species_ids, snapshots, runtime)."""
    core = register_copasi()
    t0 = _time.perf_counter()

    proc = cfg['class'](config=cfg['proc_cfg'], core=core)
    state0 = proc.initial_state()

    species_ids = proc.species_ids
    n_steps = cfg['n_steps']
    interval = cfg['proc_cfg']['time']

    snapshots = []
    for _i in range(n_steps):
        result = proc.update(
            {'species_concentrations': {}, 'species_counts': {}},
            interval=interval,
        )
        snapshots.append({
            'species_concentrations': dict(result['species_concentrations']),
            'fluxes': dict(result['fluxes']),
            'time': result['time'],
        })

    runtime = _time.perf_counter() - t0
    return species_ids, snapshots, runtime


def run_steady_state(cfg):
    """Run a CopasiSteadyStateStep — returns (species_ids, results, runtime)."""
    core = register_copasi()
    t0 = _time.perf_counter()

    proc = cfg['class'](config=cfg['proc_cfg'], core=core)
    state0 = proc.initial_state()
    result = proc.update({'species_concentrations': {}, 'counts': {}})
    runtime = _time.perf_counter() - t0

    species_ids = proc.species_ids
    return species_ids, result['results'], runtime


# ── Bigraph diagram ──────────────────────────────────────────────────

def generate_bigraph_image(cfg, species_ids):
    """Render a colored bigraph-viz PNG (as base64) for the document."""
    try:
        from bigraph_viz import plot_bigraph
    except ImportError:
        return ''

    sample_species = species_ids[:3]
    outputs = {
        sid: ['stores', 'species_concentrations', sid] for sid in sample_species
    }

    proc_type = 'step' if cfg['mode'] != 'process' else 'process'
    doc = {
        'copasi': {
            '_type': proc_type,
            'address': f'local:{cfg["class"].__name__}',
            'config': {'model_source': _MODEL_PATH},
            'inputs': {},
            'outputs': outputs,
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'inputs': {
                sid: ['stores', 'species_concentrations', sid]
                for sid in sample_species
            },
        },
    }

    node_colors = {
        ('copasi',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }

    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc,
        out_dir=outdir,
        filename='bigraph',
        file_format='png',
        remove_process_place_edges=True,
        rankdir='LR',
        node_fill_colors=node_colors,
        node_label_size='16pt',
        port_labels=False,
        dpi='150',
    )
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


def build_pbg_document(cfg):
    """Return the PBG composite document dict for JSON display."""
    if cfg['mode'] == 'step':
        return make_copasi_utc_document(
            model_source=_MODEL_PATH,
            time=cfg['proc_cfg']['time'],
            n_points=cfg['proc_cfg']['n_points'],
        )
    elif cfg['mode'] == 'process':
        return make_copasi_utc_process_document(
            model_source=_MODEL_PATH,
            time=cfg['proc_cfg']['time'],
            intervals=cfg['proc_cfg']['intervals'],
        )
    else:
        return make_copasi_steady_state_document(model_source=_MODEL_PATH)


# ── Color scheme ─────────────────────────────────────────────────────

COLOR_SCHEMES = {
    'indigo':  {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669'},
    'rose':    {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48'},
}

SPECIES_COLORS = [
    '#6366f1', '#10b981', '#f43f5e', '#f59e0b', '#06b6d4',
    '#8b5cf6', '#ec4899', '#14b8a6', '#eab308', '#3b82f6',
]


# ── HTML generation ──────────────────────────────────────────────────

def _html_escape(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def generate_html(sim_results, output_path):
    sections_html = []
    all_js_data = {}

    for idx, (cfg, data, runtime) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        mode = cfg['mode']

        if mode == 'step':
            species_ids, columns, times, values, _ = data
            # Build time-series: {column: [val, ...]}
            col_to_idx = {c: i for i, c in enumerate(columns)}
            species_series = {}
            for spid in species_ids:
                if spid in col_to_idx:
                    ci = col_to_idx[spid]
                    species_series[spid] = [row[ci] for row in values]
                else:
                    species_series[spid] = []
            n_species = len(species_ids)
            n_snaps = len(times)
            chart_times = times

        elif mode == 'process':
            species_ids, snapshots, _ = data
            n_species = len(species_ids)
            n_snaps = len(snapshots)
            # Use last time point from each snapshot for x-axis
            chart_times = [
                snap['time'][-1] if snap['time'] else i
                for i, snap in enumerate(snapshots)
            ]
            species_series = {
                spid: [snap['species_concentrations'].get(spid, 0.0) for snap in snapshots]
                for spid in species_ids
            }

        else:  # steady_state
            species_ids, results, _ = data
            n_species = len(species_ids)
            n_snaps = 1
            chart_times = [0.0]
            species_series = {
                spid: results.get('species_concentrations', {}).get(spid, [0.0])
                for spid in species_ids
            }

        all_js_data[sid] = {
            'times': chart_times,
            'species': species_series,
            'species_ids': species_ids,
            'color': cs['primary'],
            'species_colors': SPECIES_COLORS,
        }

        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg, species_ids)
        bigraph_html = (
            f'<img src="{bigraph_img}" alt="Bigraph architecture diagram">'
            if bigraph_img else
            '<em>bigraph-viz not installed; skipping diagram</em>'
        )

        pbg_doc = build_pbg_document(cfg)

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Species</span><span class="metric-value">{n_species}</span></div>
        <div class="metric"><span class="metric-label">Mode</span><span class="metric-value">{mode.replace('_', '-')}</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{n_snaps:,}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.2f}s</span></div>
      </div>

      <h3 class="subsection-title">Species Trajectories</h3>
      <div class="chart-box-full"><div id="chart-species-{sid}" class="chart-wide"></div></div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap">{bigraph_html}</div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Composite Document</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""
        sections_html.append((section, sid, pbg_doc))

    nav_items = ''.join(
        f'<a href="#sim-{cfg["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[cfg["color_scheme"]]["primary"]};">'
        f'{cfg["title"]}</a>'
        for cfg, _, __ in sim_results
    )

    pbg_docs = {sid: doc for _, sid, doc in sections_html}
    sections_html_str = ''.join(s for s, _, __ in sections_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>COPASI &times; process-bigraph Simulation Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}
.page-header {{
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:720px; }}
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100;
        flex-wrap:wrap; }}
.nav-link {{ padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             color:#334155; transition:all .15s; background:#fff; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:820px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.25rem; font-weight:700; color:#1e293b; }}
.chart-box-full {{ background:#f8fafc; border:1px solid #e2e8f0;
                   border-radius:10px; overflow:hidden; margin-bottom:1rem; }}
.chart-wide {{ height:340px; }}
.pbg-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }}
.pbg-col {{ min-width:0; }}
.bigraph-img-wrap {{ background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }}
.bigraph-img-wrap img {{ max-width:100%; height:auto; }}
.json-tree {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }}
.jt-key {{ color:#7c3aed; font-weight:600; }}
.jt-str {{ color:#059669; }}
.jt-num {{ color:#2563eb; }}
.jt-bool {{ color:#d97706; }}
.jt-null {{ color:#94a3b8; }}
.jt-toggle {{ cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }}
.jt-toggle:hover {{ color:#1e293b; }}
.jt-collapsed {{ display:none; }}
.jt-bracket {{ color:#64748b; }}
.footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }}
@media(max-width:900px) {{
  .pbg-row {{ grid-template-columns:1fr; }}
  .sim-section,.page-header,.nav {{ padding:1.5rem; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>COPASI &times; process-bigraph</h1>
  <p>The Elowitz-Leibler Repressilator (BIOMD0000000012) simulated three ways
  — one-shot UTC Step, incremental UTC Process, and steady-state solve —
  each wrapped as a <strong>process-bigraph</strong> Step or Process via
  <strong>COPASI / basico</strong>. Demonstrates SBML model composition
  in bigraph composites.</p>
</div>

<div class="nav">{nav_items}</div>

{sections_html_str}

<div class="footer">
  Generated by <strong>viva-copasi</strong> &mdash;
  COPASI + process-bigraph &mdash;
  SBML model execution in bigraph composites
</div>

<script>
const DATA = {json.dumps(all_js_data)};
const DOCS = {json.dumps(pbg_docs, indent=2)};

// ─── JSON Tree Viewer ───
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') {{
    const short = obj.length > 120 ? obj.slice(0, 120) + '…' : obj;
    return '<span class="jt-str">"' + short.replace(/</g,'&lt;').replace(/\\n/g,'\\\\n') + '"</span>';
  }}
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => {{ html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; }});
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }}
  if (typeof obj === 'object') {{
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{{}}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {{
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    }});
    html += '</div><span class="jt-bracket">}}</span>';
    return html;
  }}
  return String(obj);
}}
function toggleJt(id) {{
  const el = document.getElementById(id);
  if (!el) return;
  if (el.classList.contains('jt-collapsed')) {{
    el.classList.remove('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangledown;';
  }} else {{
    el.classList.add('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangleright;';
  }}
}}
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

// ─── Plotly Charts ───
const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:55, r:20, t:40, b:45 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

Object.keys(DATA).forEach(sid => {{
  const d = DATA[sid];
  const speciesTraces = d.species_ids.map((spid, i) => ({{
    x: d.times, y: d.species[spid],
    type: 'scatter', mode: 'lines',
    line: {{ color: d.species_colors[i % d.species_colors.length], width: 1.8 }},
    name: spid,
  }}));
  Plotly.newPlot('chart-species-' + sid, speciesTraces, {{
    ...pLayout,
    title:{{ text:'Species Concentrations vs Time', font:{{ size:12, color:'#334155' }} }},
    xaxis:{{ ...pLayout.xaxis, title:{{ text:'Time', font:{{ size:10 }} }} }},
    yaxis:{{ ...pLayout.yaxis, title:{{ text:'Concentration', font:{{ size:10 }} }} }},
    legend:{{ font:{{ size:10 }}, bgcolor:'rgba(255,255,255,0.6)' }}, showlegend:true,
  }}, pCfg);
}});
</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


def run_demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    sim_results = []

    cfg = CONFIGS[0]  # UTC Step
    print(f'Running: {cfg["title"]}...')
    t0 = _time.perf_counter()
    species_ids, columns, times, values, runtime = run_utc_step(cfg)
    sim_results.append((cfg, (species_ids, columns, times, values, runtime), runtime))
    print(f'  Runtime: {runtime:.3f}s, timepoints: {len(times)}, species: {len(species_ids)}')

    cfg = CONFIGS[1]  # UTC Process
    print(f'Running: {cfg["title"]}...')
    species_ids, snapshots, runtime = run_utc_process(cfg)
    sim_results.append((cfg, (species_ids, snapshots, runtime), runtime))
    print(f'  Runtime: {runtime:.3f}s, steps: {len(snapshots)}, species: {len(species_ids)}')

    cfg = CONFIGS[2]  # Steady State
    print(f'Running: {cfg["title"]}...')
    species_ids, results, runtime = run_steady_state(cfg)
    sim_results.append((cfg, (species_ids, results, runtime), runtime))
    print(f'  Runtime: {runtime:.3f}s, species: {len(species_ids)}')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)

    # Auto-open in Safari on macOS
    try:
        subprocess.run(['open', '-a', 'Safari', output_path], check=False)
    except Exception:
        pass


if __name__ == '__main__':
    run_demo()
