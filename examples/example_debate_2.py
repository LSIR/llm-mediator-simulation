"""Example script to run a debate simulation on nuclear energy."""

import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
# ... rest of the imports

from dotenv import load_dotenv

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.metrics.metrics_handler import MetricsHandler
from llm_mediator_simulation.metrics.perspective_api import PerspectiveScorer
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
perspective_key =os.getenv("PERSPECTIVE_API_KEY") or ""


mediator_model = GPTModel(api_key=gpt_key, model_name="gpt-4o")


# Path to the extracted model directory
model_mistral_path = "ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit"

# Initialize the MistralLocalModel with the extracted model path
debater_model = MistralLocalModel(model_name="/mnt/datastore/models/mistralai/Mistral-7B-Instruct-v0.2" ,max_length=200, debug=True, json=True)


#debater_model = GPTModel(api_key=gpt_key, model_name="gpt-4o")
#mediator_model = GoogleModel(api_key=google_key, model_name="gemini-1.5-pro")




# Debater participants
debaters = [
    DebaterConfig(
        name="Bob",
        position=DebatePosition.AGAINST,
        personalities= None,
        # personalities={
        #     PersonalityAxis.CIVILITY: AxisPosition.NEUTRAL,  # Very toxic
        #     PersonalityAxis.POLITENESS: AxisPosition.NEUTRAL,  # Very rude
        #     PersonalityAxis.EMOTIONAL_STATE: AxisPosition.NEUTRAL,  # Very angry
        #     PersonalityAxis.POLITICAL_ORIENTATION: AxisPosition.VERY_LEFT,  # Very conservative
        # },
    ),
    DebaterConfig(
        name="Alice",
        position=DebatePosition.FOR,
        personalities= None,
        # personalities={
        #     PersonalityAxis.CIVILITY: AxisPosition.NEUTRAL,  # Very civil
        #     PersonalityAxis.POLITENESS: AxisPosition.NEUTRAL,  # Very kind
        #     PersonalityAxis.EMOTIONAL_STATE: AxisPosition.NEUTRAL,  # Very calm
        #     PersonalityAxis.POLITICAL_ORIENTATION: AxisPosition.VERY_RIGHT,  # Very liberal
        # },
    ),
]

metrics = MetricsHandler(
    perspective = PerspectiveScorer(api_key=perspective_key),
    model=mediator_model,
    argument_qualities=[
        ArgumentQuality.APPROPRIATENESS,
        ArgumentQuality.CLARITY,
        ArgumentQuality.LOCAL_ACCEPTABILITY,
        ArgumentQuality.EMOTIONAL_APPEAL,
        ArgumentQuality.GLOBAL_RELEVANCE,
        
    ],
)  

# The conversation summary handler (keep track of the general history and of the n latest messages)
summary_config = SummaryConfig(latest_messages_limit=3, debaters=debaters)

# The debate configuration (which topic to discuss, and customisable instructions)
debate_config = DebateConfig(
    statement="Abortion should be legal",
)

mediator_config = None #MediatorConfig() #None

# The debate runner
debate = DebateHandler(
    debater_model=debater_model,
    mediator_model=mediator_model,
    debaters=debaters,
    config=debate_config,
    summary_config=summary_config,
    metrics_handler=metrics,
    mediator_config=mediator_config,
    relative_memory = True,
)

debate.run(rounds=4)

debate.pickle("debateNuclearMediator")