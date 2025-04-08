"""Example script to run a debate simulation on nuclear energy."""

import json
import os
import time

from dotenv import load_dotenv

from llm_mediator_simulation.models.gpt_models import GPTModel
from llm_mediator_simulation.models.ollama_local_server_model import OllamaLocalModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debate.handler import DebateHandler
from llm_mediator_simulation.simulation.summary.config import SummaryConfig
from llm_mediator_simulation.visualization.transcript import debate_transcript

load_dotenv()

SEED = 42

# TODO HF server, ollama pretrained model and seed

gpt_key = os.getenv("GPT_API_KEY") or ""
mediator_model = GPTModel(api_key=gpt_key, model_name="gpt-4o")

# debater_model = OllamaLocalModel(model_name="mistral-nemo")
debater_model = OllamaLocalModel(model_name="olmo2:13b")


debaters = []

# The conversation summary handler (keep track of the general history and of the n latest messages)
summary_config = SummaryConfig(latest_messages_limit=3, debaters=debaters, ignore=True)

submission_id = "3ko9vd"
comment_id = "cuz35v2"

with open("data/reddit/cmv/statements.json", "r", encoding="utf-8") as f:
    statements = json.load(f)
    statement = statements[submission_id]
# statement = "The anti-SJW movement has become a cesspool of bigotry"
# Remove trailing dot from the statement
if statement.endswith("."):
    statement = statement[:-1]

# TODO Check prompt

# The debate configuration (which topic to discuss, and customisable instructions)
debate_config = DebateConfig(
    statement=statement,
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
)

truncated_chat_path = f"data/reddit/cmv/sub_{submission_id}-comment_{comment_id}.csv"
debate.preload_csv_chat(
    "data/reddit/cmv/sub_3ko9vd-comment_cuz35v2.csv", app="reddit", truncated_num=2
)

debate.run(rounds=1)

name_timestamp = time.strftime("%Y%m%d-%H%M%S")
output_path = "debates_sandbox"
data = debate.to_debate_pickle()
print(debate_transcript(data))
debate.pickle(os.path.join(output_path, f"debate_{name_timestamp}"))
