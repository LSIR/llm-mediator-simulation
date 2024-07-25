"""Mistral local-running model wrapper"""

from typing import Literal, override

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)

# Taken from the HuggingFace api
FEW_SHOT_PREPROMPT = """User: What is yout favorite condiment?
Assistant: I don't have a favorite condiment as I don't consume food or condiments. However, I can tell you that some common favorite condiments among people include ketchup, mayonnaise, hot sauce, mustard, and soy sauce. These condiments can add flavor, texture, and enhancement to various dishes.

User:"""

JSON_FEW_SHOT_PREPROMPT = """User: Do you want to add a message to the conversation?

Answer in JSON format with the following structure only:
```json
{
    "do_intervene": bool,
    "intervention_justification": a string justification of why you want to intervene or not,
    "text": the text message for your intervention. Leave empty if you decide not to intervene
}
```
Assistant:```json
{
    "do_intervene": true,
    "intervention_justification": "I think it is important to add a comment to the debate to clarify a point.",
    "text": "I think it is important to clarify that the data presented in the previous message is outdated and no longer accurate."
}
```
User:"""


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
        debug: bool = False,
    ):
        """Initialize a Mistral model.

        Args:
            model_name: Mistral model name.
            max_length: Maximum token length of the generated text.
            num_return_sequences: Number of generated sentences.
            temperature: Sampling temperature.
            top_p: Top-p sampling ratio.
            do_sample: Whether to sample or not.
            debug: Displays verbose prompts and responses.
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
        self.debug = debug

    @override
    def sample(self, prompt: str) -> str:

        prompt = f"{JSON_FEW_SHOT_PREPROMPT}{prompt}\nAssistant: "

        if self.debug:
            print("Prompt:")
            print("----------------------")
            print(prompt)
            print()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids.to("cuda"),
                attention_mask=inputs.attention_mask,
                max_length=self.max_length,
                num_return_sequences=self.num_return_sequences,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.pad_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if self.debug:
            print("Response:")
            print("---------------------")
            print(generated_text)
            print()

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

        prompts = [f"{FEW_SHOT_PREPROMPT}{prompt}\nAssistant: " for prompt in prompts]

        inputs = self.tokenizer(prompts, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids.to("cuda"),
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
