from datetime import datetime
import os
import sys
import torch
from dotenv import load_dotenv
import jsonlines
from dataclasses import dataclass
from typing import List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import random
import numpy as np
import torch.distributed as dist
from accelerate import Accelerator
import re

# TODO Should we add the full context, i.e., more previous comments?
# TODO Should we include personalities at training time?
# Don't include few-shot at training time.

# TODO run.AI and fine-tune 32B 

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic algorithms for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Fine-tune in mixed precision FP16 in RTX 8000 and bfloat16 in RunAI's GPUs

load_dotenv()

# Ensure WANDB_API_KEY is set
if not os.getenv("WANDB_API_KEY"):
    raise ValueError("Please set your WANDB_API_KEY environment variable")


@dataclass
class RedditComment:
    id: str
    userid: str
    text: str
    timestamp: int

    def __post_init__(self):
        # Process the text field with regex substitution
        self.text = re.sub(r">\s*(.*?)\n", r"You said: '\1'\n", self.text)

@dataclass
class RedditPair:
    statement: str
    submission_id: str
    penultimate_utterance: RedditComment
    last_utterance: RedditComment
    split: str

def load_reddit_pairs(file_path: str) -> List[RedditPair]:
    pairs = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            pairs.append(RedditPair(
                statement=obj['statement'],
                submission_id=obj['submission_id'],
                penultimate_utterance=RedditComment(**obj['penultimate_utterance']),
                last_utterance=RedditComment(**obj['last_utterance']),
                split=obj['split']
            ))
    return pairs

def format_prompt(pair: RedditPair) -> str:
    return f"""You are roleplaying this real person:
username: {pair.last_utterance.userid}

Act as a Reddit user engaged in a conversation about the following statement: "{pair.statement}".

Here is the last comment:
- {pair.penultimate_utterance.userid}: {pair.penultimate_utterance.text}


Based on the other participant's opinions relatively to yours, as expressed in the conversation so far, post a new comment.
Do not repeat yourself and do not quote other participants.
Your new comment:
- {pair.last_utterance.userid}:""" # if you specify "post a new comment of maximum 4 sentences" the model tends to generated a 4-length numbered list...
# TODO: move prompt to config

def format_completion(pair: RedditPair) -> str:
    return f" {pair.last_utterance.text}"  # Add a space at the start to ensure consistent tokenization

def prepare_data(pairs: List[RedditPair], split: str) -> Dataset:
    data = []
    for pair in pairs:
        if pair.split == split:
            prompt = format_prompt(pair)
            completion = format_completion(pair)
            data.append({
                "prompt": prompt,
                "completion": completion
            })
    return Dataset.from_list(data)

class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_examples=1, accelerator=None):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_examples = num_examples
        self.accelerator = accelerator
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        # Only run on main process
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return

        # Select random examples from validation set
        indices = torch.randperm(len(self.eval_dataset))[:self.num_examples].tolist()
        
        print("\n" + "="*50)
        print(f"Evaluation at epoch {state.epoch:.2f}")
        print("="*50)
        
        for idx in indices:
            example = self.eval_dataset[idx]
            prompt = example["prompt"]
            completion = example["completion"]
            
            # Generate text
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Print to terminal
            print("\nPrompt:")
            print("-"*20)
            prompt_len = len(prompt)
            print(prompt)
            print("\nGround Truth:")
            print("-"*20)
            print(completion)
            print("\nGenerated:")
            print("-"*20)
            print(generated_text[prompt_len:])
            print("\n" + "="*50)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    try:
        # Set random seed for reproducibility
        set_seed(cfg.seed)

        # Get Hydra output directory
        output_dir = HydraConfig.get().runtime.output_dir
        timestamp = output_dir.split('/')[-1]  # Get the timestamp from the output directory

        # Initialize accelerator with device placement
        accelerator = Accelerator(
            device_placement=True,
            mixed_precision='bf16' if torch.cuda.is_bf16_supported() else 'fp16'
        )

        # Initialize wandb only on the main process
        if accelerator.is_main_process:
            wandb.init(
                project="olmo-reddit-finetune",
                name=f"{cfg.training.name}-{timestamp}",
                dir=output_dir
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.path,
            trust_remote_code=cfg.model.trust_remote_code
        )

        assert tokenizer.padding_side == "right"
        assert tokenizer.pad_token

        # Load and prepare data
        pairs = load_reddit_pairs(cfg.data.path)
        train_dataset = prepare_data(pairs, "train")
        val_dataset = prepare_data(pairs, "val")

        if torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16 # bfloat16 Not supported in RTX 8000
        else:
            torch_dtype = torch.float32 # Change to float16 if too much memory is required
        
        # Load model in 4-bit
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.path,
            trust_remote_code=cfg.model.trust_remote_code,
            load_in_4bit=cfg.model.load_in_4bit,
            torch_dtype=torch_dtype,
        )

        # Prepare model for training (only for quantization training)
        # model = prepare_model_for_kbit_training(model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=list(cfg.lora.target_modules),
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type
        )

        # Create PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Training arguments
        training_args = SFTConfig(
            output_dir=output_dir,
            run_name=cfg.training.name,
            num_train_epochs=cfg.training.num_train_epochs,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            learning_rate=cfg.training.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=cfg.training.logging_steps,
            eval_steps=cfg.training.eval_steps,
            save_steps=cfg.training.save_steps,
            warmup_steps=cfg.training.warmup_steps,
            report_to=cfg.training.report_to,
            seed=cfg.seed,
            eval_strategy=cfg.training.eval_strategy,
            optim=cfg.training.optim,
            weight_decay=cfg.training.weight_decay,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            ddp_find_unused_parameters=False
        )

        # Create Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[GenerationCallback(tokenizer, val_dataset, accelerator=accelerator)],
        )

        # # Prepare the trainer with accelerator
        trainer = accelerator.prepare(trainer)

        # Train
        trainer.train()
        
        # Save the final model
        trainer.save_model(f"{output_dir}/final") 
        
        # Close wandb only on the main process
        if accelerator.is_main_process:
            wandb.finish()

    finally:
        # Clean up distributed process group
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    config = OmegaConf.load(os.getenv("CONFIG_PATH", "configs") + "/config.yaml")
    output_path = config.training.output_path

    # if output_path does not exist, change it to /home/laugier/olmo2-cga-cmv/sft (for runai, saving on PVC)
    if not os.path.exists(output_path):
        output_path = "/home/laugier/olmo2-cga-cmv/sft"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    sys.argv.append(f"hydra.run.dir={output_path}/{timestamp}")
    main() 