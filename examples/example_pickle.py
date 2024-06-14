"""Read a pickled conversation and display its metrics."""

from rich import print as rprint

from llm_mediator_simulations.simulation.debate import Debate

data = Debate.unpickle("debate.pkl")


rprint(data)
