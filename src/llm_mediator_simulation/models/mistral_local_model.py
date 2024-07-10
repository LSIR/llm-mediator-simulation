"""Mistral local-running model wrapper"""

from typing import override

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_mediator_simulation.models.language_model import LanguageModel


class MistralLocalModel(LanguageModel):
    """Mistral local-running model wrapper"""

    def __init__(self):
        """Initialize a Mistral model.

        Args:
            model_name: Mistral model name.
        """

        self.model_name = "mistralai/Mistral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto"
        )

    @override
    def sample(self, prompt: str) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=50,  # TODO: adapt this
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text
