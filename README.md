# Neural Network Simulation & Visualization GUI

A 2-day hackathon MVP: a small GUI tool for **defining**, **simulating**, and
**visually exploring** simple recurrent neural networks. You set network and
simulation parameters, press **Run**, and see the network as a graph —
**blue** edges for positive (excitatory) connections, **red** for negative
(inhibitory), with line **thickness** proportional to connection strength.
Nodes can optionally be colored/sized by their simulated activity.

## Quick start

```bash
pip install nnsimviz      # from PyPI
nnsimviz                  # launches the Streamlit GUI
```

Or from source (editable install with dev tools):

```bash
pip install -e ".[dev]"
nnsimviz                  # or: python -m streamlit run src/nnsimviz/app.py
```

Then set parameters in the sidebar and press **▶ Run simulation**.

Run the test suite:

```bash
pytest                    # 54 tests
pytest --cov=nnsimviz     # with coverage
```

## Architecture

The project is modular, with a strict one-way data flow and a shared data
contract. The GUI is a thin orchestrator — it holds **no** simulation or
graph-building logic.

```
 user widgets
      │
      ▼
 ProjectConfig            (configs.py)   ── dataclasses + validation
      │
      ▼
 weight matrix W          (models.py)    ── model library → NumPy W
      │
      ▼
 SimulationResult         (simulation.py)── recurrent dynamics over time
      │
      ▼
 Plotly figure            (visualization.py) ── NetworkX layout + 2-trace plot
      │
      ▼
 display + export         (app.py / io_utils.py)
```

### Files

| File | Responsibility | Depends on |
|------|----------------|------------|
| `configs.py` | All parameter dataclasses (`NetworkConfig`, `SimulationConfig`, `VisualizationConfig`, `ProjectConfig`) + validation. The single source of truth for data shapes. | — |
| `models.py` | Model library. `RandomWeightedNetwork` turns a `NetworkConfig` into a signed, weighted NumPy matrix `W`. Registry lets the GUI list models. | `configs` |
| `simulation.py` | `Simulator` integrates `x[t+1] = x[t] + dt·(−x[t] + W·tanh(x[t]) + input) + noise` and returns a `SimulationResult`. | `configs` |
| `visualization.py` | `NetworkVisualizer` (`build_graph`, `compute_layout`, `create_figure`) → interactive Plotly graph. No GUI imports. | `configs`, `simulation` |
| `io_utils.py` | Save/load config (JSON), export weights & activity (CSV). | `configs`, `simulation` |
| `app.py` | Streamlit GUI. Reads widgets → builds `ProjectConfig` → calls model → simulation → visualization → displays figure + summary + exports. | all of the above |

### Shared data contract

All modules agree on these shapes (defined in `configs.py` / `simulation.py`):

- **`NetworkConfig`**: `n_neurons`, `connection_probability`, `weight_scale`,
  `positive_connection_ratio`, `model_type`, `random_seed`
- **`SimulationConfig`**: `duration`, `dt`, `input_type`, `input_amplitude`,
  `noise_level`
- **`VisualizationConfig`**: `layout_type`, `show_labels`, `edge_width_scale`,
  `min_edge_abs_weight`, `node_size_scale`, `show_activity_on_nodes`
- **`ProjectConfig`**: `network`, `simulation`, `visualization`
- **`SimulationResult`**: `time`, `activity` (neurons × timesteps),
  `final_state`, `weight_matrix`, `config`, `metadata`

> Note: `W[i, j]` is the weight of the connection **from neuron j to neuron i**,
> so the dynamics can use `W @ x` directly.

## Visualization approach

Uses the standard two-trace Plotly + NetworkX pattern: one set of `Scatter`
line traces for edges (segments separated by `None`), one `Scatter` marker
trace for nodes, positioned by `spring_layout` or `circular_layout`. On top of
that, edges are split by sign into blue/red traces and bucketed by magnitude so
line width reflects `|weight|`. Reference:
<https://plotly.com/python/network-graphs/>

## What works (Milestone 0 → improvements)

- ✅ End-to-end pipeline: parameters → graph, runs without crashing
- ✅ Blue/red signed edges, thickness by strength
- ✅ Two layouts (spring, circular)
- ✅ Weak-edge filtering, activity-on-nodes, hover info, legend
- ✅ Validation with friendly error messages
- ✅ Export: config (JSON), weights (CSV), activity (CSV)

## Adding a new model

Implement the `NetworkModel` protocol in `models.py` (a `generate(config) → W`
method plus `name`/`description`/`equation`) and register it in
`MODEL_REGISTRY`. The GUI picks it up automatically — no other file changes.

## Demo script

1. Start with defaults, press **Run** → see the network graph.
2. Increase **connection probability** → graph gets denser.
3. Increase **weight scale** → edges get thicker.
4. Toggle **Color nodes by activity** → nodes reflect simulated state.
