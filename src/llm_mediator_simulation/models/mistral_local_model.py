"""Mistral local-running model wrapper"""

from typing import override

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

        self.model_name = "mistralai/Mistral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto"
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
                max_length=self.max_length,
                num_return_sequences=self.num_return_sequences,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
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

        self.model_name = "mistralai/Mistral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto"
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
                max_length=self.max_length,
                num_return_sequences=self.num_return_sequences,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )

        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return generated_texts
