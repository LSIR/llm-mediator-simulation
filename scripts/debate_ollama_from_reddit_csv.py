"""Example script to run a debate simulation on nuclear energy."""

import json
import os

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from llm_mediator_simulation.models.gpt_models import GPTModel
from llm_mediator_simulation.models.ollama_local_server_model import OllamaLocalModel
from llm_mediator_simulation.simulation.debate.handler import DebateHandler
from llm_mediator_simulation.visualization.transcript import debate_transcript


@hydra.main(version_base=None, config_path="configs", config_name="reddit_cmv")
def main(config):
    load_dotenv()

    seed = config.seed

    # TODO HF server: olmo pretrained model and seed

    gpt_key = os.getenv("GPT_API_KEY") or ""
    mediator_model = GPTModel(api_key=gpt_key, model_name="gpt-4o")

    debater_model = OllamaLocalModel(model_name="mistral-nemo")
    # debater_model = OllamaLocalModel(model_name="olmo2:13b")

    debaters = []

    # The conversation summary handler (keep track of the general history and of the n latest messages)
    summary_config = instantiate(config.summary_config)

    submission_id = "3ko9vd"
    comment_id = "cuz35v2"

    with open("data/reddit/cmv/statements.json", "r", encoding="utf-8") as f:
        statements = json.load(f)
        statement = statements[submission_id]

    # Remove trailing dot from the statement
    if statement.endswith("."):
        statement = statement[:-1].strip()

    # The debate configuration (which topic to discuss, and customisable instructions)
    debate_config = instantiate(config.debate_config, statement=statement)

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
        seed=seed,
    )

    truncated_chat_path = (
        f"data/reddit/cmv/sub_{submission_id}-comment_{comment_id}.csv"
    )
    debate.preload_csv_chat(truncated_chat_path, app="reddit", truncated_num=2)

    debate.run(rounds=1)

    data = debate.to_debate_pickle()
    print(debate_transcript(data))
    debate.pickle(os.path.join(HydraConfig.get().runtime.output_dir, "debate"))


if __name__ == "__main__":
    main()
