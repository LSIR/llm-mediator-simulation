"""Example script to run a debate simulation on nuclear energy."""

from datetime import time
import os

from dotenv import load_dotenv

from llm_mediator_simulation.models.google_models import GoogleModel
from llm_mediator_simulation.models.mistral_local_model import MistralLocalModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debate.handler import DebateHandler
from llm_mediator_simulation.simulation.debater.config import (
    AxisPosition,
    DebatePosition,
    DebaterConfig,
    PersonalityAxis,
)
from llm_mediator_simulation.simulation.summary.config import SummaryConfig

load_dotenv()

# Load the mediator model
google_key = os.getenv("VERTEX_AI_API_KEY") or ""
mediator_model = GoogleModel(api_key=google_key, model_name="gemini-1.5-pro")

debater_model = MistralLocalModel(model_name="/mnt/datastore/models/mistralai/Mistral-7B-Instruct-v0.2" ,
                                  max_length=200, 
                                  debug=True, 
                                  json=True)

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

# The conversation summary handler (keep track of the general history and of the n latest messages)
summary_config = SummaryConfig(latest_messages_limit=3, debaters=debaters)

# The debate configuration (which topic to discuss, and customisable instructions)
debate_config = DebateConfig(
    statement="We should use nuclear power.",
)

# mediator_config = MediatorConfig()

# The debate runner
debate = DebateHandler(
    debater_model=debater_model,
    mediator_model=mediator_model,
    debaters=debaters,
    config=debate_config,
    summary_config=summary_config,
    metrics_handler=None,
    mediator_config=None,
    seed=42,
    variable_personality=False,
)

debate.run(rounds=3)

name_timestamp = time.strftime('%Y%m%d-%H%M%S')
debate.pickle(f"../debates_sandbox/debate_{name_timestamp}")
