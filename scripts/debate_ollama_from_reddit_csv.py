"""Example script to run a debate simulation on nuclear energy.
Safe-to_ignore warnings from Tensorflow: https://github.com/tensorflow/tensorflow/issues/62075
"""

import json
import os
import sys
from datetime import datetime

import httpx
import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from natsort import natsorted
from omegaconf import OmegaConf

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
    # Note For conv submission_id: "3mhgci" ; comment_id: "cveysuq" -> The last two authors are new to the conv... model users from their past comments?
    # OK Create a train-dev-test-splitted dataset with convs with < 6 messages to create A) Few shot examples and B) a fine-tuning dataset.
    # Done 1 Prompt engineering
    # OK Remove json format. Note, if we keep the json format, olmo2 pretrained very often refuses to intervene.
    # OK Few shot examples from CMV convs not in the simulation dataset
    # OK for now (Try Mistral, LlaMa, Olmo1 Pretrained to whether Olmo2 pretraining is flawed. If yes, then no other choice than fine-tuning.)
    # Done 2 Model exploration
    # OK investigate why pretrained Olmo2 always output the same low quality responses.
    # OK look for the most toxic-like LLMs (Mistral and others ?)
    # What is in omlo 2 post training exactly? OK
    # Abliterated models: https://huggingface.co/collections/failspy/abliterated-v3-664a8ad0db255eefa7d0012b
    # https://huggingface.co/mlabonne/Daredevil-8B-abliterated
    # We don't know exactly what is in Mistral's post training
    # Done 2 Fine-tuning ++
    # -> Finetuning Olmo pre or post-trained on Custom "Conversations going Awry-like" Reddit data? -> CMV convs with < 6 messages. Those that have 4 messages
    # Questions. Ablation 3
    # # If we fine-tune, should we add instrucitons in the input?
    # # Should we add few shot examples at training time or at inference time?
    # # SFT or RL ?

    # Done 3 Generate personalities based on all previous messages of the Reddit user?
    # TODO 4 Lina's improvements: personalized summary.
    # TODO IDEA: Use the Wikipedia version of Conv gone awry focusing on Controversial Wiki Talk pages (Israel, feminism, etc.). Question, is OlMo2 pretrained on wiki talk pages?

    # 0. Zero-shot
    # 1. Few shot (RAG Few-shot?)
    # 2. Generated personality
    # 3. Personalized summary (TODO)
    # + Combined
    # 4. Fine-tuning
    # + Combined

    gpt_key = os.getenv("GPT_API_KEY") or ""
    mediator_model = GPTModel(api_key=gpt_key, model_name="gpt-4o")

    # OlMo2 post-trained (SFT, DPO, and Instruct) were fine-tuned with safety filters so let's stay with the pretrained model

    if not config.json_debater_reponse:
        stop_strings = ["\n"]
    else:
        stop_strings = None

    debater_model = HFLocalServerModel(
        port=PORT,
        max_new_tokens=config.max_new_tokens,
        debug=True,
        repetition_penalty=config.repetition_penalty,
        stop_strings=stop_strings,  # ["\n-", "\n -"],
    )

    debaters = []

    # The conversation summary handler (keep track of the general history and of the n latest messages)
    summary_config = instantiate(config.summary_config)

    split = config.split
    conversations_path = f"data/reddit/cmv/{split}/"

    with open("data/reddit/cmv/statements.json", "r", encoding="utf-8") as f:
        statements = json.load(f)

    if config.few_shot_samples:
        with open("data/reddit/cmv/few_shot_samples.jsonl", "r") as f:
            lines = f.readlines()
            few_shot_samples = [json.loads(line) for line in lines]
    else:
        few_shot_samples = None

    for truncated_chat_path in natsorted(os.listdir(conversations_path)):
        assert truncated_chat_path.endswith(".csv")
        submission_id = truncated_chat_path.split("-")[0].split("_")[1]
        comment_id = truncated_chat_path.split("-")[1].split(".")[0].split("_")[1]
        statement = statements[submission_id]

        # Remove trailing dot from the statement
        if statement.endswith("."):
            statement = statement[:-1].strip()

        # Remove starting "CMV:" from the statement
        if statement.startswith("CMV:"):
            statement = statement[4:].strip()

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
            json_debater_reponse=config.json_debater_reponse,
            few_shot_samples=few_shot_samples,
        )

        debate.preload_csv_chat(
            f"data/reddit/cmv/{split}/{truncated_chat_path}",
            app="reddit",
            truncated_num=2,
            load_debater_profiles=config.load_debater_profiles,
            debater_profiles_path="data/reddit/cmv/reddit_user_profiles.json",
            prune_debaters=config.prune_debaters,
        )

        debate.run(rounds=1)

        data = debate.to_debate_pickle()
        transcript = debate_transcript(data)
        print(transcript)

        output_path = HydraConfig.get().runtime.output_dir

        # save the debate to a file
        debate_path = os.path.join(output_path, "debates")

        if not os.path.exists(debate_path):
            os.makedirs(debate_path)

        debate_file = os.path.join(
            debate_path,
            f"sub_{submission_id}-comment_{comment_id}",
        )

        debate.pickle(debate_file)

        # save the transcript to a file
        transcript_path = os.path.join(output_path, "transcripts")
        if not os.path.exists(transcript_path):
            os.makedirs(transcript_path)

        transcript_file = os.path.join(
            transcript_path,
            f"sub_{submission_id}-comment_{comment_id}.txt",
        )

        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript)


if __name__ == "__main__":
    model_name = httpx.get(
        f"http://localhost:{PORT}/model_name",
        timeout=40,
    ).text
    sys.argv.append(f"+debater_model_name={model_name}")

    quantization = httpx.get(
        f"http://localhost:{PORT}/model_quantization",
        timeout=40,
    ).text

    if quantization:
        sys.argv.append(f"+debater_quantization={quantization}")

    output_dir_name = []

    # Save results in outputs/{split}/json_fs_profiles/{timestamp}

    # loag hydra config from scripts/configs/reddit_cmv.yaml without hydra, just as a simple yaml file
    config = OmegaConf.load("scripts/configs/reddit_cmv.yaml")

    if config.json_debater_reponse:
        output_dir_name.append("json")
    else:
        output_dir_name.append("nojson")

    if config.few_shot_samples:
        output_dir_name.append("fs")
    else:
        output_dir_name.append("nofs")

    if config.load_debater_profiles:
        output_dir_name.append("profiles")
    else:
        output_dir_name.append("noprofiles")

    output_dir = "_".join(output_dir_name)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    split = config.split

    sys.argv.append(f"hydra.run.dir=outputs/{split}/{output_dir}/{timestamp}")

    main()
