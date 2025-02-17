"""Example script to run parallel debate simulations on nuclear energy.

BUG: Google async models may raise an error when deallocated at the end due to this bug. It is harmless.
-> https://github.com/google-gemini/generative-ai-python/issues/207#issuecomment-2308952931
"""

import asyncio
import os

from dotenv import load_dotenv

from llm_mediator_simulation.metrics.async_metrics_handler import AsyncMetricsHandler
from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.models.google_models import AsyncGoogleModel
from llm_mediator_simulation.simulation.debate.async_handler import AsyncDebateHandler
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debater.config import (
    AxisPosition,
    DebatePosition,
    DebaterConfig,
    PersonalityAxis,
)
from llm_mediator_simulation.simulation.mediator.config import MediatorConfig
from llm_mediator_simulation.simulation.summary.config import SummaryConfig

load_dotenv()


gpt_key = os.getenv("GPT_API_KEY") or ""
google_key = os.getenv("VERTEX_AI_API_KEY") or ""
perspective_key = os.getenv("PERSPECTIVE_API_KEY") or ""

# mediator_model = AsyncGPTModel(api_key=gpt_key, model_name="gpt-4o")
# debater_model = BatchedMistralLocalModel(model_name="/mnt/datastore/models/mistralai/Mistral-7B-Instruct-v0.2" ,max_length=200, debug=True, json=True)
mediator_model = AsyncGoogleModel(api_key=google_key, model_name="gemini-1.5-pro")

PARALLEL_DEBATES = 2

# Debater participants
debaters = [
    DebaterConfig(
        name="Alice",
        position=DebatePosition.AGAINST,
        personalities={
            PersonalityAxis.CIVILITY: AxisPosition.VERY_RIGHT,  # Very toxic
            PersonalityAxis.POLITENESS: AxisPosition.VERY_RIGHT,  # Very rude
            PersonalityAxis.EMOTIONAL_STATE: AxisPosition.VERY_RIGHT,  # Very angry
            PersonalityAxis.POLITICAL_ORIENTATION: AxisPosition.VERY_LEFT,  # Very conservative
        },
    ),
    DebaterConfig(
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

metrics = AsyncMetricsHandler(
    model=mediator_model,
    argument_qualities=[
        ArgumentQuality.APPROPRIATENESS,
        ArgumentQuality.CLARITY,
        ArgumentQuality.LOCAL_ACCEPTABILITY,
        ArgumentQuality.EMOTIONAL_APPEAL,
    ],
)  # perspective=PerspectiveScorer(api_key=perspective_key))


# The conversation summary handler (keep track of the general history and of the n latest messages)
summary_config = SummaryConfig(latest_messages_limit=3, debaters=debaters)

# The debate configuration (which topic to discuss, and customisable instructions)
debate_config = DebateConfig(
    statement="We should use nuclear power.",
)


mediator_config = MediatorConfig()


debate = AsyncDebateHandler(
    debater_model=mediator_model,
    mediator_model=mediator_model,
    debaters=debaters,
    config=debate_config,
    summary_config=summary_config,
    metrics_handler=metrics,
    mediator_config=mediator_config,
    parallel_debates=PARALLEL_DEBATES,
)


asyncio.run(debate.run(rounds=3))


debate.pickle("async_debate")
