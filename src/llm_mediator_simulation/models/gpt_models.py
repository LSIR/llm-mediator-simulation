"""OpenAI GPT model wrapper."""

import asyncio
from typing import Literal, override

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)


class GPTModel(LanguageModel):
    """OpenAI GPT model wrapper."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: Literal["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"],
    ):
        """Initialize a GPT model.

        Args:
            api_key: OpenAI API key.
            model_name: OpenAI model name.
        """
        self._api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    @override
    def sample(self, prompt: str) -> str:

        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]

        result = self.client.chat.completions.create(
            messages=messages, model=self.model_name, n=1
        )
        content = result.choices[0].message.content

        return content if content else ""
    
    @override
    def generate_response(self, prompt:str) -> str:
        self.sample(prompt)
        


class AsyncGPTModel(AsyncLanguageModel):
    """OpenAI GPT model wrapper."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: Literal["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"],
    ):
        """Initialize a GPT model.

        Args:
            api_key: OpenAI API key.
            model_name: OpenAI model name.
        """
        self._api_key = api_key
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key)

    @override
    async def sample(self, prompts: list[str]) -> list[str]:
        # Prepare prompts
        messages: list[list[ChatCompletionMessageParam]] = [
            [{"role": "user", "content": prompt}] for prompt in prompts
        ]

        # Await all completions asynchronously
        results = await asyncio.gather(
            *[
                self.client.chat.completions.create(
                    messages=message, model=self.model_name, n=1
                )
                for message in messages
            ]
        )

        contents = [result.choices[0].message.content or "" for result in results]

        return contents
