# LLM Mediator Simulations

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3120/">
    <img src="https://img.shields.io/badge/python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000?style=for-the-badge&logo=black&logoColor=white" alt="Code style: black">
  </a>
</p>

Simulate online debates using LLMs. This project runs on Python 3.12.

**WARNING**: as this project deals with toxicity simulation, you may encounter shocking language in prompt configurations source code.
[See the full warning](./WARNING.md).

See the [overview](/docs/OVERVIEW.md) for more details.

## Project Structure

```bash
├── examples                  # Example scripts
│
├── llm_mediator_simulations  # Main package
│   │
│   ├── metrics     # Metrics computations
│   ├── models      # LLM model wrappers
│   ├── simulation  # Debate simulation handler classes
│   └── utils       # Utilities
```

## Get Started

This project was setup using [hatch](https://hatch.pypa.io/latest/).

```bash
pip install --user hatch
hatch env create
source venv/bin/activate  # Activate the python virtual environment
pip install -e .  # Install the current package
```

## Examples

Example scripts are located in the [`examples/`](./examples) folder.

```bash
python examples/example_async.py
```

Run a local LLM with the `example_server.py` script.
Analyze generated debates with the `example_analysis.py` script.
