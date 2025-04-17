"""Example script to run a debate simulation on nuclear energy.
Safe-to_ignore warnings from Tensorflow: https://github.com/tensorflow/tensorflow/issues/62075
"""

import json
import os
import sys

import httpx
import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from llm_mediator_simulation.models.gpt_models import GPTModel
from llm_mediator_simulation.models.hf_local_server_model import (
    HFLocalServerModel,
)
from llm_mediator_simulation.simulation.debate.handler import DebateHandler
from llm_mediator_simulation.visualization.transcript import debate_transcript

PORT = 8000


@hydra.main(version_base=None, config_path="configs", config_name="reddit_cmv")
def main(config):
    load_dotenv()

    seed = config.seed

    # olmo 2 13B pretrained refuses to intervene and always output the same justification
    # olmo 2 13B post-trained and olmo 2 32B post-trained quantized float 16 intervene but is too polite
    # Mistral 7B pretrained does not intervene and returns empty justification
    # Mistral 7B post-trained intervenes
    # TODO 1 Prompt engineering
    # -> Remove json format. Note, if we keep the json format, olmo2 pretrained very often refuses to intervene.
    # -> Few shot examples from CMV convs with < 6 messages.
    # OK for now (Try Mistral, LlaMa, Olmo1 Pretrained to whether Olmo2 pretraining is flawed. If yes, then no other choice than fine-tuning.)
    # 2 Model exploration
    # OK investigate why pretrained Olmo2 always output the same low quality responses.
    # OK look for the most toxic-like LLMs (Mistral and others ?)
    # What is in omlo 2 post training exactly? OK
    # Abliterated models: https://huggingface.co/collections/failspy/abliterated-v3-664a8ad0db255eefa7d0012b
    # https://huggingface.co/mlabonne/Daredevil-8B-abliterated
    # We don't know exactly what is in Mistral's post training
    # TODO 2 Fine-tuning ++
    # -> Finetuning Olmo pre or post-trained on Custom "Conversations going Awry-like" Reddit data? -> CMV convs with < 6 messages. Those that have 4 messages
    # Questions. Ablation 3
    # # If we fine-tune, should we add instrucitons in the input?
    # # Should we add few shot examples at training time or at inference time?
    # # SFT or RL ?

    # TODO 3 Generate personalities based on all previous messages of the Reddit user?
    # TODO 4 Lina's improvements: personalized summary.
    # TODO IDEA: Use the Wikipedia version of Conv gone awry focusing on Controversial Wiki Talk pages (Israel, feminism, etc.). Question, is OlMo2 pretrained on wiki talk pages?

    # TODO Compare
    # 1. Few shot
    # 2. Generated personality
    # 3. Personalized summary
    # + Combined
    # 4. Fine-tuning
    # + Combined

    gpt_key = os.getenv("GPT_API_KEY") or ""
    mediator_model = GPTModel(api_key=gpt_key, model_name="gpt-4o")

    # OlMo2 post-trained (SFT, DPO, and Instruct) were fine-tuned with safety filters so let's stay with the pretrained model

    # debater_model = HFLocalModel(
    #     model_name="allenai/OLMo-2-1124-13B",
    #     max_new_tokens=config.max_new_tokens,
    #     json=True,
    # )

    debater_model = HFLocalServerModel(port=PORT, max_new_tokens=config.max_new_tokens)

    # debater_model = OllamaLocalModel(model_name="mistral-nemo")
    # debater_model = OllamaLocalModel(model_name="olmo2:13b")

    debaters = []

    # The conversation summary handler (keep track of the general history and of the n latest messages)
    summary_config = instantiate(config.summary_config)

    submission_id = config.submission_id
    comment_id = config.comment_id

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
    debate_path = os.path.join(HydraConfig.get().runtime.output_dir, "debate")
    debate.pickle(debate_path)

    # printable_data = data.to_printable()
    # console = Console(force_terminal=True, record=True)
    # pprint(printable_data, console=console)


if __name__ == "__main__":
    model_name = httpx.get(
        f"http://localhost:{PORT}/model_name",
        timeout=40,
    ).text
    sys.argv.append(f"+debater_model_name={model_name}")
    main()
