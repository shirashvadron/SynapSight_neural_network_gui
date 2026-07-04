# Neural Network Simulation & Visualization GUI

A Streamlit-based visualization tool for building, importing, simulating, and exploring small recurrent neural networks.

The project lets a user define a network, run either continuous dynamics or event-based spike-like propagation, and inspect the result through interactive graph views, activity plots, animations, PCA projections, and spike rasters.

Positive connections are shown in blue, negative connections in red, and edge thickness represents connection strength.

---

## Current features

- Generate recurrent neural networks from several built-in models:
  - random weighted network
  - excitatory/inhibitory network
  - symmetric weighted network
  - modular network
- Import an external weight matrix from:
  - `.csv`
  - `.npy`
  - `.npz`
- Run two simulation modes:
  - continuous recurrent dynamics
  - event-based / spike-like threshold propagation
- Visualize:
  - network graph
  - activity over time
  - animated network activity
  - animated PCA state-space trajectory
  - spike raster for event-based simulations
- Export:
  - config as JSON
  - weight matrix as CSV
  - activity matrix as CSV
  - graph image from the Plotly toolbar / export button
- Test coverage for configs, models, simulations, event simulation, integration, visualization, and IO utilities.

---

## Installation from source

Clone the repository, enter the project folder, then install in editable mode:

```bash
pip install -e ".[dev]"
```

This installs the package and development tools such as `pytest`.

---

## Run the app

After installation:

```bash
nnsimviz
```

Or run Streamlit directly:

```bash
streamlit run src/nnsimviz/app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## Run tests

If the project is installed with:

```bash
pip install -e ".[dev]"
```

then run:

```bash
pytest
```

If you did not install the package, run tests from the project root with:

```bash
PYTHONPATH=src pytest
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
pytest
```

The current project version has 135 passing tests.

---

## Project structure

```text
neural_network_gui/
├── examples/
│   └── default_config.json
├── src/
│   └── nnsimviz/
│       ├── app.py
│       ├── cli.py
│       ├── configs.py
│       ├── event_simulation.py
│       ├── help_texts.py
│       ├── io_utils.py
│       ├── models.py
│       ├── pipeline.py
│       ├── simulation.py
│       └── visualization.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Main modules

| File | Responsibility |
|---|---|
| `configs.py` | Dataclasses and validation for network, simulation, event, visualization, and project settings. |
| `models.py` | Built-in network generators and model registry. |
| `simulation.py` | Continuous recurrent simulation engine. |
| `event_simulation.py` | Event-based / spike-like threshold propagation engine. |
| `pipeline.py` | Streamlit-independent orchestration: build/import network, choose simulator, return result. |
| `visualization.py` | Plotly and NetworkX visualizations: graph, activity, animation, PCA, raster. |
| `io_utils.py` | Config JSON, CSV export, and imported weight matrix loading/validation. |
| `help_texts.py` | Sidebar tooltip text for GUI controls. |
| `app.py` | Streamlit user interface. |
| `cli.py` | Console command entry point for launching the Streamlit app. |

---

## Network models

### Random weighted network

Each directed pair of neurons may be connected with probability `connection_probability`.

Each existing edge is independently assigned a positive or negative sign according to `positive_connection_ratio`.

### Excitatory/inhibitory network

Each source neuron has a fixed identity.

If source neuron `j` is excitatory, all outgoing weights from `j` are positive.  
If source neuron `j` is inhibitory, all outgoing weights from `j` are negative.

### Symmetric weighted network

Connections are reciprocal:

```text
W[i, j] = W[j, i]
```

This is useful for Hopfield-like or energy-based network experiments.

### Modular network

Neurons are split into modules.

Connections inside the same module use `connection_probability`.  
Connections between different modules use `inter_module_probability`.

---

## Matrix convention

The project uses the following convention:

```text
W[i, j] = connection from source neuron j to target neuron i
```

This means the continuous simulation can compute recurrent input as:

```python
W @ x
```

Example:

```text
W[3, 1] = 0.8
```

means:

```text
neuron 1 activates neuron 3 with weight 0.8
```

This convention is especially important when importing an external weight matrix.

---

## Continuous simulation

The continuous simulator evolves the network state over time using recurrent dynamics.

Simplified form:

```text
dx/dt = -x + W @ tanh(x) + input
```

Supported integration methods:

- Euler
- Heun / improved Euler
- RK4

Optional convergence detection can stop the simulation early when the state change becomes very small.

---

## Event-based simulation

The event-based simulator is a discrete threshold-propagation mode.

At each event step:

1. scheduled external or propagated events are added to neuron states
2. neurons above threshold spike
3. spiking neurons reset
4. outgoing weights schedule events for the next step
5. optional decay is applied

This is useful for visualizing spike-like signal propagation through a network.

Important note: this is not a full biological LIF neuron model. It is a discrete threshold-reset propagation model.

---

## Imported networks

The GUI can import a square weight matrix from:

- `.csv`
- `.npy`
- `.npz`

For `.npz`, the file should contain either:

- an array named `W`, or
- exactly one array

Imported matrices must be:

- 2D
- square
- numeric
- finite

When an imported matrix is used, it overrides the generated model and updates `n_neurons` automatically.

---

## Visualization

The app currently provides:

- interactive network graph
- activity over time
- animated network activity
- PCA trajectory animation
- event spike raster
- event log table

Graph styling:

- blue edges = positive weights
- red edges = negative weights
- edge thickness = absolute weight magnitude
- weak edges can be hidden with `min_edge_abs_weight`
- node color/size can reflect final activity

---

## Example config

See:

```text
examples/default_config.json
```

The config contains four sections:

```text
network
simulation
event
visualization
```

The `event` section is included even when `simulation_type` is `"continuous"`, so the same config file can easily be switched to event mode.

---

## Known limitations / next improvements

The current project is functional, but several improvements would make it stronger:

- event-based simulation should store spike trains separately from post-reset activity
- event-based PCA should not show continuous fixed-point markers
- GUI should expose custom external event lists
- directed graph edges should show arrow direction
- imported matrices should have an optional transpose checkbox
- self-connections should either be visualized or explicitly rejected
- large networks need better decluttering tools
- README, examples, and tests should stay updated whenever config fields change

---

## Development workflow

Recommended workflow for changes:

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

After editing:

```bash
pytest
git status
git add .
git commit -m "Describe your change"
git push -u origin feature/your-feature-name
```

Then open a pull request on GitHub.

---

## License

MIT