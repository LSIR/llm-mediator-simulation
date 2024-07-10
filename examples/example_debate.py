"""Example script to run a debate simulation on nuclear energy."""

import os
import pickle

from dotenv import load_dotenv

from llm_mediator_simulation.metrics.metrics_handler import MetricsHandler
from llm_mediator_simulation.metrics.perspective_api import PerspectiveScorer
from llm_mediator_simulation.models.gpt_models import GPTModel
from llm_mediator_simulation.simulation.configuration import (
    DebateConfig,
    DebatePosition,
    Debater,
    Mediator,
    Personality,
)
from llm_mediator_simulation.simulation.debate import Debate, Debater
from llm_mediator_simulation.simulation.summary_handler import SummaryHandler
from llm_mediator_simulation.utils.decorators import BENCHMARKS, print_benchmarks

load_dotenv()

gpt_key = os.getenv("GPT_API_KEY") or ""
perspective_key = os.getenv("PERSPECTIVE_API_KEY") or ""

model = GPTModel(api_key=gpt_key, model_name="gpt-4o")


# Debater participants
debaters = [
    Debater(
        name="Alice",
        position=DebatePosition.AGAINST,
        personality=[
            Personality.TOXIC,
            Personality.AGGRESSIVE,
            Personality.INFORMAL,
            Personality.INSULTING,
        ],
    ),
    Debater(
        name="Bob",
        position=DebatePosition.FOR,
    ),
]

metrics = MetricsHandler(perspective=PerspectiveScorer(api_key=perspective_key))

# The conversation summary handler (keep track of the general history and of the n latest messages)
summary = SummaryHandler(model=model, latest_messages_limit=1)

# The debate configuration (which topic to discuss, and customisable instructions)
configuration = DebateConfig(
    statement="We should use nuclear power.",
)

mediator = Mediator()

# The debate runner
debate = Debate(
    model=model,
    debaters=debaters,
    configuration=configuration,
    summary_handler=summary,
    metrics_handler=metrics,
    mediator=mediator,
)

debate.run(rounds=3)

print_benchmarks()

debate.pickle("debate.pkl")
pickle.dump(BENCHMARKS, open("benchmarks.pkl", "wb"))
