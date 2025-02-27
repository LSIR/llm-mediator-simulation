"""Example script to run a debate simulation on nuclear energy."""

import os
import time

from dotenv import load_dotenv

from llm_mediator_simulation.models.gpt_models import GPTModel
from llm_mediator_simulation.models.ollama_local_server_model import OllamaLocalModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debate.handler import DebateHandler
from llm_mediator_simulation.simulation.summary.config import SummaryConfig
from llm_mediator_simulation.visualization.transcript import debate_transcript
from scripts.default_debaters import debaters

load_dotenv()

SEED = 42

gpt_key = os.getenv("GPT_API_KEY") or ""
mediator_model = GPTModel(api_key=gpt_key, model_name="gpt-4o")

debater_model = OllamaLocalModel(model_name="mistral")


# The conversation summary handler (keep track of the general history and of the n latest messages)
summary_config = SummaryConfig(latest_messages_limit=3, debaters=debaters, ignore=True)

# The debate configuration (which topic to discuss, and customisable instructions)
debate_config = DebateConfig(
    statement="We should use nuclear power.",
)

mediator_config = None  # MediatorConfig()

metrics = None

# The debate runner
debate = DebateHandler(
    debater_model=debater_model,
    mediator_model=mediator_model,
    debaters=debaters,
    config=debate_config,
    summary_config=summary_config,
    metrics_handler=metrics,
    mediator_config=mediator_config,
    seed=SEED,
    variable_personality=False,
)

debate.run(rounds=2)

name_timestamp = time.strftime("%Y%m%d-%H%M%S")
output_path = "debates_sandbox"
data = debate.to_debate_pickle()
print(debate_transcript(data))
debate.pickle(os.path.join(output_path, f"debate_{name_timestamp}"))
