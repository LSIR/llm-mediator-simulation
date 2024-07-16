# TODO : test async debate, with only 1 debate and only gpt api.

# 1 debate, 1 gpt model, no metrics
# 1 debate, 1 gpt model, metrics + perspective
# 2 debates, 1 gpt model, metrics + perspective

# tester l'output pickle


import asyncio
import os

from dotenv import load_dotenv

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.metrics.metrics_handler import AsyncMetricsHandler
from llm_mediator_simulation.metrics.perspective_api import PerspectiveScorer
from llm_mediator_simulation.models.gpt_models import AsyncGPTModel
from llm_mediator_simulation.simulation.async_debate import AsyncDebate
from llm_mediator_simulation.simulation.configuration import (
    DebateConfig,
    DebatePosition,
    Debater,
    Mediator,
    Personality,
)
from llm_mediator_simulation.simulation.summary_handler import AsyncSummaryHandler

load_dotenv()


gpt_key = os.getenv("GPT_API_KEY") or ""
perspective_key = os.getenv("PERSPECTIVE_API_KEY") or ""

mediator_model = AsyncGPTModel(api_key=gpt_key, model_name="gpt-3.5-turbo")
PARALLEL_DEBATES = 2

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

# The debate configuration (which topic to discuss, and customisable instructions)
configuration = DebateConfig(
    statement="We should use nuclear power.",
)


# The conversation summary handler (keep track of the general history and of the n latest messages)
summary = AsyncSummaryHandler(
    model=mediator_model,
    latest_messages_limit=1,
    debaters=debaters,
    parallel_debates=PARALLEL_DEBATES,
)

# Metrics
metrics = AsyncMetricsHandler(
    perspective=PerspectiveScorer(api_key=perspective_key, rate_limit=0),
    model=mediator_model,
    argument_qualities=[
        ArgumentQuality.LOCAL_ACCEPTABILITY,
        ArgumentQuality.CLARITY,
        ArgumentQuality.CREDIBILITY,
    ],
)

mediator = Mediator()

debate = AsyncDebate(
    debater_model=mediator_model,
    mediator_model=mediator_model,
    debaters=debaters,
    configuration=configuration,
    summary_handler=summary,
    metrics_handler=metrics,
    mediator=mediator,
    parallel_debates=PARALLEL_DEBATES,
)


asyncio.run(debate.run(rounds=1))


debate.pickle("async_debate")
