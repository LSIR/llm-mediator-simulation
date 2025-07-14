"""Mistral local-running model wrapper"""

import os
from typing import Any, Literal, override

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)
from llm_mediator_simulation.utils.reproducibility import set_transformers_seed


class HFLocalModel(LanguageModel):
    """HuggingFace's local-running model wrapper"""

    def __init__(
        self,
        *,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        quantization: Literal["4_bits"] | None = None,
        debug: bool = False,
        json: bool = False,
        **kwargs: dict,
    ):
        """Initialize a HuggingFace model.

        Args:
            model_name: Model name, or path to such a model.
            max_new_tokens: Maximum newly generated tokens
            num_return_sequences: Number of generated sentences.
            temperature: Sampling temperature.
            top_p: Top-p sampling ratio.
            do_sample: Whether to sample or not.
            quantization: BitsAndBytes precision.
            debug: Displays verbose prompts and responses.
            json: Whether to enforce JSON generation.
            kwargs: Additional arguments for the model.

        Recommendations can be found in Google Prompt Engineering White Paper:
        https://drive.google.com/file/d/1AbaBYbEa_EbPelsT40-vj64L-2IwUJHy/view
        """
        self.model_path = model_name
        custom_model = False
        # If lora fine-tuned
        if os.path.exists(model_name) and "adapter_model.safetensors" in os.listdir(
            model_name
        ):
            peft_config = PeftConfig.from_pretrained(model_name)
            self.model_name = peft_config.base_model_name_or_path
            custom_model = True

        else:
            self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=QUANTIZATION_CONFIG[quantization],
            # revision="stage1-step721901-tokens6056B",
            **kwargs,
        )

        # If lora fine-tuned
        if custom_model:
            self.model = PeftModel.from_pretrained(model, model_name)

        else:
            self.model = model
        # Parameters
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.debug = debug
        self.json = json

        if quantization is None:
            self.quantization = "no quantization"
        else:
            self.quantization = quantization

    @override
    def sample(self, prompt: str, seed: int | None = None, **kwargs: Any) -> str:
        for parameter in [
            "max_new_tokens",
            "num_return_sequences",
            "temperature",
            "top_k",
            "top_p",
            "do_sample",
        ]:
            if parameter not in kwargs:
                kwargs[parameter] = getattr(self, parameter)

        json = kwargs.pop("json", self.json)

        stop_strings = kwargs.pop("stop_strings", None)
        assert not (json and stop_strings), (
            "stop_strings and json cannot be used together. "
            "Please use one or the other."
        )

        debug = kwargs.pop("debug", self.debug)

        if json:
            prompt = f"{prompt}```json"  #

        if debug:
            print("Prompt:")
            print("----------------------")
            print(prompt)
            print()

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Seeding
        if seed is not None:
            set_transformers_seed(seed)  # sampling tokens generation time

        assert (
            inputs.input_ids.shape[1] + kwargs["max_new_tokens"]
        ) < self.tokenizer.model_max_length, (
            "Prompt too long for the model. Please reduce the number of tokens."
        )

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids.to("cuda"),
                attention_mask=inputs.attention_mask,
                # num_return_sequences=self.num_return_sequences,
                # temperature=self.temperature,
                # top_k=self.top_k,
                # top_p=self.top_p,
                # do_sample=self.do_sample,
                stop_strings=(["```"] if json else stop_strings),
                tokenizer=self.tokenizer,
                **kwargs,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if debug:
            print("Response:")
            print("---------------------")
            print(generated_text[len(prompt) :])
            print()

        return generated_text


# TODO Check if async really needed here?
# If no, then make a single model class to handle both cases
# If Yes, make a superclass that both classes will inherit from
class BatchedHFLocalModel(AsyncLanguageModel):
    """HuggingFace's local-running model wrapper, in a batched async-compatible version."""

    def __init__(
        self,
        *,
        max_length: int = 50,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        quantization: Literal["4_bits"] | None = None,
        json: bool = False,
    ):
        """Initialize a HuggingFace model.

        Args:
            model_name: Mistral model name.
            max_length: Maximum token length of the generated text.
            num_return_sequences: Number of generated sentences.
            temperature: Sampling temperature.
            top_p: Top-p sampling ratio.
            do_sample: Whether to sample or not.
            json: Whether to enforce JSON generation.
        """

        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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
        self.json = json

    @override
    async def sample(
        self, prompts: list[str], seed: int | None = None, **kwargs: Any
    ) -> list[str]:
        preprompt = JSON_FEW_SHOT_PREPROMPT if self.json else FEW_SHOT_PREPROMPT
        postprompt = "Assistant:```json" if self.json else "Assistant: "

        prompts = [f"{preprompt}{prompt}{postprompt}" for prompt in prompts]

        inputs = self.tokenizer(prompts, return_tensors="pt")

        # Seeding
        if seed is not None:
            set_transformers_seed(seed)  # sampling tokens generation time

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
                # Stop conditions
                stop_strings=["```"] if self.json else ["User:"],
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_length,
            )

        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return generated_texts


# Quantization configs
from transformers import BitsAndBytesConfig  # noqa: E402

# 4 bit precision
config_4bits = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # bfloat16 Not supported in RTX 8000
)

# https://huggingface.co/blog/4bit-transformers-bitsandbytes
# "A rule of thumb is:
# - use double quant if you have problems with memory,
# - use NF4 for higher precision,
# - and use a 16-bit dtype for faster finetuning.
# [...]
# Among GPUs, there should not be any hardware requirement about this method,
# therefore any GPU could be used to run the 4bit quantization as long as you have CUDA>=11.2 installed.
# Keep also in mind that the computation is not done in 4bit, the weights and activations are compressed
# to that format and the computation is still kept in the desired or native dtype.
# [...]
# Similarly as the integration of LLM.int8 presented in this blogpost the integration heavily relies on the accelerate library.
# Therefore, any model that supports accelerate loading (i.e. the device_map argument when calling from_pretrained)
# should be quantizable in 4bit.
# At this time of writing, the models that support accelerate are:
# [
#     'bigbird_pegasus', 'blip_2', 'bloom', 'bridgetower', 'codegen', 'deit', 'esm',
#     'gpt2', 'gpt_bigcode', 'gpt_neo', 'gpt_neox', 'gpt_neox_japanese', 'gptj', 'gptsan_japanese',
#     'lilt', 'llama', 'longformer', 'longt5', 'luke', 'm2m_100', 'mbart', 'mega', 'mt5', 'nllb_moe',
#     'open_llama', 'opt', 'owlvit', 'plbart', 'roberta', 'roberta_prelayernorm', 'rwkv', 'switch_transformers',
#     't5', 'vilt', 'vit', 'vit_hybrid', 'whisper', 'xglm', 'xlm_roberta'
# ]""
#
# https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes?bnb=4-bit
# "Quantize a model by passing a BitsAndBytesConfig to from_pretrained().
# This works for any model in any modality, as long as it supports Accelerate and contains torch.nn.Linear layers."


QUANTIZATION_CONFIG = {
    None: None,
    "4_bits": config_4bits,
}
