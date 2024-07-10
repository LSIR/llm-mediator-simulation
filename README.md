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
```

## Examples

Example scripts are located in the [`examples/`](./examples) folder.
Run them using the `python -m examples.<script>` command to properly resolve the `llm_mediator_simulations` package.

```bash
python -m examples.example_debate
```
