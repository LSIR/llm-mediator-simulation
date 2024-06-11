"""Example script to run a debate simulation on nuclear energy."""

import os

from dotenv import load_dotenv

from llm_mediator_simulations.models.gpt_models import GPTModel
from llm_mediator_simulations.simulation.debate import (
    Debate,
    DebatePosition,
    Debater,
    Personality,
)
from llm_mediator_simulations.simulation.summary import Summary

load_dotenv()

gpt_key = os.getenv("GPT_API_KEY") or ""

model = GPTModel(api_key=gpt_key, model_name="gpt-4o")


debaters = [
    Debater(
        position=DebatePosition.FOR,
        personality=[Personality.ANGRY, Personality.INSULTING, Personality.LIBERAL],
    ),
    Debater(
        position=DebatePosition.AGAINST,
        personality=[Personality.AGGRESSIVE, Personality.CONSERVATIVE],
    ),
]

summary = Summary(model=model, latest_messages_limit=1)

debate = Debate(
    model=model,
    statement="We should use nuclear power.",
    debaters=debaters,
    summary_handler=summary,
)

debate.run(rounds=3)

print()
print("Debate messages:")
print("----------------")

for message in debate.messages:
    print(message[0], "-", message[1])
    print()
