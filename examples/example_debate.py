"""Example script to run a debate simulation on nuclear energy."""

import os

from dotenv import load_dotenv

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.metrics.metrics_handler import MetricsHandler
from llm_mediator_simulation.models.google_models import GoogleModel
from llm_mediator_simulation.models.gpt_models import GPTModel
from llm_mediator_simulation.models.mistral_local_model import MistralLocalModel



from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debate.handler import DebateHandler
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

#mediator_model = GPTModel(api_key=gpt_key, model_name="gpt-4o")
debater_model = MistralLocalModel(model_name="/mnt/datastore/models/mistralai/Mistral-Small-Instruct-2409" ,max_length=200, debug=True, json=True, quantization = "4_bits")
mediator_model = GoogleModel(api_key=google_key, model_name="gemini-1.5-pro")


# Debater participants
debaters = [
    DebaterConfig(
        name="Alice",
        position=DebatePosition.AGAINST,
        personalities={
            PersonalityAxis.CIVILITY: AxisPosition.VERY_RIGHT,  # Very civil
            PersonalityAxis.POLITENESS: AxisPosition.VERY_RIGHT,  # Very kind
            PersonalityAxis.EMOTIONAL_STATE: AxisPosition.VERY_RIGHT,  # Very calm
            PersonalityAxis.POLITICAL_ORIENTATION: AxisPosition.VERY_LEFT,  # Very concervative
        },
    ),
    DebaterConfig(
        name="Bob",
        position=DebatePosition.FOR,
        personalities={
           PersonalityAxis.CIVILITY: AxisPosition.RIGHT,  # Very civil
            PersonalityAxis.POLITENESS: AxisPosition.RIGHT,  # Very kind
            PersonalityAxis.EMOTIONAL_STATE: AxisPosition.RIGHT,  # Very calm
            PersonalityAxis.POLITICAL_ORIENTATION: AxisPosition.VERY_LEFT,  # Very concervati
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
)  
#perspective= PerspectiveScorer(api_key=perspective_key)

# The conversation summary handler (keep track of the general history and of the n latest messages)
summary_config = SummaryConfig(latest_messages_limit=3, debaters=debaters)

# The debate configuration (which topic to discuss, and customisable instructions)
debate_config = DebateConfig(
    statement="We should use nuclear power.",
)

mediator_config = MediatorConfig()

# The debate runner
debate = DebateHandler(
    debater_model=mediator_model,
    mediator_model=mediator_model,
    debaters=debaters,
    config=debate_config,
    summary_config=summary_config,
    metrics_handler=metrics,
    mediator_config=mediator_config,
)

debate.run(rounds=2)

debate.pickle("debate8")

