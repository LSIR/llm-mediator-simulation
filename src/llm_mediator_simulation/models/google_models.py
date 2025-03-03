"""Google Vertex AI model wrapper."""

import asyncio
from typing import Literal, override

import google.generativeai as genai
from google.generativeai.types import (
    GenerationConfigType,
    HarmBlockThreshold,
    HarmCategory,
)

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)


class GoogleModel(LanguageModel):
    """Google Vertex AI model wrapper."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: Literal["gemini-1.0-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
        harm_block_threshold: HarmBlockThreshold = HarmBlockThreshold.BLOCK_NONE,
        temperature: float = 1,
        seed: int | None = None,
    ):
        """Initialize a Google Vertex AI model.

        Args:
            api_key: OpenAI API key.
            model_name: OpenAI model name.
            harm_block_threshold: Harm block threshold.
            temperature: the model temperature.
            seed: Seeding sampling at generation time. Currently not avalaible through the GenerativeAI Python SDK.
        """
        genai.configure(api_key=api_key)

        config: GenerationConfigType = {
            "temperature": temperature,
        }

        # https://github.com/google-gemini/generative-ai-python/issues/605
        # Currently seeding Gemin not available through the GenerativeAI Python SDK
        if seed is not None:
            raise NotImplementedError(
                "Seed not available through the GenerativeAI Python SDK"
            )
        #    config["seed"] = seed

        self.model = genai.GenerativeModel(model_name, generation_config=config)

        self.safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": harm_block_threshold,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": harm_block_threshold,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": harm_block_threshold,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": harm_block_threshold,
            },
        ]

    @override
    def sample(self, prompt: str, seed: int | None = None) -> str:
        """Generate text based on the given prompt."""

        response = self.model.generate_content(
            prompt,
            safety_settings=self.safety_settings,  # type: ignore
        )

        return response.text


class AsyncGoogleModel(AsyncLanguageModel):
    """Google Vertex AI model wrapper."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: Literal["gemini-1.0-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
        harm_block_threshold: HarmBlockThreshold = HarmBlockThreshold.BLOCK_NONE,
    ):
        """Initialize a Google Vertex AI model.

        Args:
            api_key: OpenAI API key.
            model_name: OpenAI model name.
            harm_block_threshold: Harm block threshold.
        """
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model_name)

        self.safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": harm_block_threshold,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": harm_block_threshold,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": harm_block_threshold,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": harm_block_threshold,
            },
        ]

    @override
    async def sample(self, prompts: list[str], seed: int | None = None) -> list[str]:
        """Generate text based on the given prompt."""

        results = await asyncio.gather(
            *[
                self.model.generate_content_async(
                    prompt, safety_settings=self.safety_settings
                )
                for prompt in prompts
            ]
        )

        return [result.text for result in results]
