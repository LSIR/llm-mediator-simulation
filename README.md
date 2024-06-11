# LLM Mediator Simulations

<div style="text-align: center;">
  [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
</div>

Simulate online debates using LLMs. This project runs on Python 3.12.

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

You must ensure that Python 3.12 is available on your machine.

```bash
python --version  # Python 3.12.x
python -m venv venv  # Create a virtual environment
source venv/bin/activate  # Activate the python virtual environment
pip install -r requirements.txt  # Install the dependencies
```

## Examples

Example scripts are located in the [`examples/`](./examples) folder.
Run them using the `python -m examples.<script>` command to properly resolve the `llm_mediator_simulations` package.

```bash
python -m examples.example_debate
```
