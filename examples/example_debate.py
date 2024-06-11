"""Example script to run a debate simulation on nuclear energy."""

import os

from dotenv import load_dotenv

from llm_mediator_simulations.models.gpt_models import GPTModel
from llm_mediator_simulations.simulation.configuration import (
    DebateConfig,
    DebatePosition,
    Debater,
)
from llm_mediator_simulations.simulation.debate import Debate, DebatePosition, Debater
from llm_mediator_simulations.simulation.summary import Summary

load_dotenv()

gpt_key = os.getenv("GPT_API_KEY") or ""

model = GPTModel(api_key=gpt_key, model_name="gpt-4o")


# Debater participants
debaters = [
    Debater(
        position=DebatePosition.FOR,
    ),
    Debater(
        position=DebatePosition.FOR,
    ),
]

# The conversation summary handler (keep track of the general history and of the n latest messages)
summary = Summary(model=model, latest_messages_limit=1)

# The debate configuration (which topic to discuss, and customisable instructions)
configuration = DebateConfig(
    statement="We should use nuclear power.",
)

# The debate runner
debate = Debate(
    model=model,
    debaters=debaters,
    configuration=configuration,
    summary_handler=summary,
)

debate.run(rounds=3)

print()
print("Debate messages:")
print("----------------")

for message in debate.messages:
    print(message[0], "-", message[1])
    print()
