"""Read a pickled conversation and display its metrics."""

import pickle

from rich import print as rprint

from llm_mediator_simulation.simulation.debate import Debate
from llm_mediator_simulation.utils.decorators import print_benchmarks
from llm_mediator_simulation.visualization.transcript import debate_transcript

data = Debate.unpickle("debate.pkl")

rprint(data)

benchmarks = pickle.load(open("benchmarks.pkl", "rb"))

print_benchmarks(benchmarks)

# Generate the transcript
transcript = debate_transcript(data)

with open("transcript.txt", "w") as f:
    f.write(transcript)
