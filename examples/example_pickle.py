"""Read a pickled conversation and display its metrics."""

import pickle

from llm_mediator_simulation.simulation.debate import Debate
from llm_mediator_simulation.utils.decorators import print_benchmarks

data = Debate.unpickle("debate.pkl")

benchmarks = pickle.load(open("benchmarks.pkl", "rb"))

print_benchmarks(benchmarks)
