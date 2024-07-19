"""Mistral local-running model wrapper"""

from typing import Literal, override

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)


class MistralLocalModel(LanguageModel):
    """Mistral local-running model wrapper"""

    def __init__(
        self,
        *,
        max_length: int = 50,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        quantization: Literal["4_bits"] | None = None,
    ):
        """Initialize a Mistral model.

        Args:
            model_name: Mistral model name.
            max_length: Maximum token length of the generated text.
            num_return_sequences: Number of generated sentences.
            temperature: Sampling temperature.
            top_p: Top-p sampling ratio.
            do_sample: Whether to sample or not.
        """

        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=QUANTIZATION_CONFIG[quantization],
        )

        # Parameters
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

    @override
    def sample(self, prompt: str) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=self.max_length,
                num_return_sequences=self.num_return_sequences,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.pad_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text


class BatchedMistralLocalModel(AsyncLanguageModel):
    """Mistral local-running model wrapper, in a batched async-compatible version."""

    def __init__(
        self,
        *,
        max_length: int = 50,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        quantization: Literal["4_bits"] | None = None,
    ):
        """Initialize a Mistral model.

        Args:
            model_name: Mistral model name.
            max_length: Maximum token length of the generated text.
            num_return_sequences: Number of generated sentences.
            temperature: Sampling temperature.
            top_p: Top-p sampling ratio.
            do_sample: Whether to sample or not.
        """

        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=QUANTIZATION_CONFIG[quantization],
        )

        # Parameters
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

    @override
    async def sample(self, prompts: list[str]) -> list[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=self.max_length,
                num_return_sequences=self.num_return_sequences,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.pad_token_id,
            )

        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return generated_texts


# Quantization configs
from transformers import BitsAndBytesConfig

# 4 bit precision
config_4bits = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

QUANTIZATION_CONFIG = {
    None: None,
    "4_bits": config_4bits,
}
