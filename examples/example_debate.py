"""Example script to run a debate simulation on nuclear energy."""

import os

from dotenv import load_dotenv

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.metrics.metrics_handler import MetricsHandler
from llm_mediator_simulation.models.google_models import GoogleModel
from llm_mediator_simulation.simulation.configuration import (
    AxisPosition,
    DebateConfig,
    DebatePosition,
    Debater,
    Mediator,
    PersonalityAxis,
)
from llm_mediator_simulation.simulation.debate import Debate, Debater
from llm_mediator_simulation.simulation.summary_handler import SummaryHandler

load_dotenv()

gpt_key = os.getenv("GPT_API_KEY") or ""
google_key = os.getenv("VERTEX_AI_API_KEY") or ""
perspective_key = os.getenv("PERSPECTIVE_API_KEY") or ""

# mediator_model = GPTModel(api_key=gpt_key, model_name="gpt-4o")
# debater_model = MistralLocalModel(model_name="/mnt/datastore/models/mistralai/Mistral-7B-Instruct-v0.2" ,max_length=200, debug=True, json=True)
mediator_model = GoogleModel(api_key=google_key, model_name="gemini-1.5-pro")


# Debater participants
debaters = [
    Debater(
        name="Alice",
        position=DebatePosition.AGAINST,
        personalities={
            PersonalityAxis.CIVILITY: AxisPosition.VERY_RIGHT,  # Very toxic
            PersonalityAxis.POLITENESS: AxisPosition.VERY_RIGHT,  # Very rude
            PersonalityAxis.EMOTIONAL_STATE: AxisPosition.VERY_RIGHT,  # Very angry
            PersonalityAxis.POLITICAL_ORIENTATION: AxisPosition.VERY_LEFT,  # Very conservative
        },
    ),
    Debater(
        name="Bob",
        position=DebatePosition.FOR,
        personalities={
            PersonalityAxis.CIVILITY: AxisPosition.VERY_LEFT,  # Very civil
            PersonalityAxis.POLITENESS: AxisPosition.VERY_LEFT,  # Very kind
            PersonalityAxis.EMOTIONAL_STATE: AxisPosition.VERY_LEFT,  # Very calm
            PersonalityAxis.POLITICAL_ORIENTATION: AxisPosition.VERY_RIGHT,  # Very liberal
        },
    ),
]

metrics = MetricsHandler(
    model=mediator_model,
    argument_qualities=[
        ArgumentQuality.APPROPRIATENESS,
        ArgumentQuality.CLARITY,
        ArgumentQuality.LOCAL_ACCEPTABILITY,
        ArgumentQuality.EMOTIONAL_APPEAL,
    ],
)  # perspective=PerspectiveScorer(api_key=perspective_key))

# The conversation summary handler (keep track of the general history and of the n latest messages)
summary = SummaryHandler(model=mediator_model, latest_messages_limit=1)

# The debate configuration (which topic to discuss, and customisable instructions)
configuration = DebateConfig(
    statement="We should use nuclear power.",
)

mediator = Mediator()

# The debate runner
debate = Debate(
    debater_model=mediator_model,
    mediator_model=mediator_model,
    debaters=debaters,
    configuration=configuration,
    summary_handler=summary,
    metrics_handler=metrics,
    mediator=mediator,
)

debate.run(rounds=3)

debate.pickle("debate")
