"""Example script to run a debate simulation on nuclear energy."""

import asyncio
import os
import time

from default_debaters import debaters
from dotenv import load_dotenv

from llm_mediator_simulation.models.gpt_models import AsyncGPTModel
from llm_mediator_simulation.models.ollama_local_server_model import (
    AsyncOllamaLocalModel,
)
from llm_mediator_simulation.simulation.debate.async_handler import AsyncDebateHandler
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.summary.config import SummaryConfig
from llm_mediator_simulation.visualization.transcript import debate_transcript

load_dotenv()

SEED = 42

gpt_key = os.getenv("GPT_API_KEY") or ""
mediator_model = AsyncGPTModel(api_key=gpt_key, model_name="gpt-4o")

PARALLEL_DEBATES = 3
debater_model = AsyncOllamaLocalModel(model_name="mistral-nemo")

# TODO Add a Github Issue to Parallelize on different:
# - debate configs (mainly debate statement)
# - debater configs

# The conversation summary handler (keep track of the general history and of the n latest messages)
summary_config = SummaryConfig(latest_messages_limit=3, debaters=debaters, ignore=True)

# The debate configuration (which topic to discuss, and customisable instructions)
debate_config = DebateConfig(
    statement="We should use nuclear power.",
)

mediator_config = None  # MediatorConfig()

metrics = None

# The debate runner
debate = AsyncDebateHandler(
    debater_model=debater_model,
    mediator_model=mediator_model,
    debaters=debaters,
    config=debate_config,
    summary_config=summary_config,
    metrics_handler=metrics,
    mediator_config=mediator_config,
    parallel_debates=PARALLEL_DEBATES,
    seed=SEED,
)

# debate.run(rounds=10)
asyncio.run(debate.run(rounds=4))

name_timestamp = time.strftime("%Y%m%d-%H%M%S")
output_path = "debates_sandbox"
data_first_debate = debate.to_first_debate_pickle()
print(debate_transcript(data_first_debate))
debate.pickle(os.path.join(output_path, f"async_debate_{name_timestamp}"))
