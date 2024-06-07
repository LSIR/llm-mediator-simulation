"""Google Vertex AI model wrapper."""

from typing import Literal, override

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from llm_mediator_simulations.models.language_model import LanguageModel


class GoogleModel(LanguageModel):
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

        self.model = genai.GenerativeModel(f"model/{model_name}")

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: harm_block_threshold,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: harm_block_threshold,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: harm_block_threshold,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: harm_block_threshold,
        }

    @override
    def sample(self, prompt: str) -> str:
        """Generate text based on the given prompt."""

        return self.model.generate_content(
            prompt, safety_settings=self.safety_settings  # type: ignore
        )
