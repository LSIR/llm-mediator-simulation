"""Read a pickled conversation and display its metrics."""

import pickle

from rich import print as rprint

from llm_mediator_simulations.simulation.debate import Debate
from llm_mediator_simulations.utils.decorators import print_benchmarks

data = Debate.unpickle("debate.pkl")

rprint(data)

benchmarks = pickle.load(open("benchmarks.pkl", "rb"))

print_benchmarks(benchmarks)
