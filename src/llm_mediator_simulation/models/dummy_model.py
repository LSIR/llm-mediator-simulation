"""Dummy model for testing purposes."""

import re
from typing import override

from llm_mediator_simulation.models.language_model import LanguageModel
from llm_mediator_simulation.utils.json import parse_llm_json


class DummyModel(LanguageModel):
    def __init__(self, model_name: str = "dummy_model") -> None:
        self.model_name = model_name

    @override
    def sample(self, prompt: str, seed: int | None = None) -> str:
        if (
            "You have the opportunity to make your personality evolve" in prompt
        ):  # For testing personality update
            # Change all the double quotes after the colon to single quotes
            # Enclose all text after colons in double quotes.
            prompt = re.sub(r'"(.*?)"(:)', r'"""\1"""\2', prompt)
            prompt = re.sub(r"(: )(.*?)(,?\n)", r'\1"""\2"""\3', prompt)
            prompt = re.sub(r'"', r"'", prompt)
            prompt = re.sub(r"'''", r'"', prompt)
            answer_format: dict[str, str] = parse_llm_json(prompt)

            str_key_update = ""
            for key, value in answer_format.items():
                dummy_update = re.search(r"'(.*?)'", value)
                assert dummy_update is not None
                dummy_update = dummy_update.group(1)
                str_key_update += f""""{key}": "{dummy_update}",
                            """
            str_key_update = str_key_update.strip()[:-1]
            dummy_response = f"""```json
                        {{
                            {str_key_update}
                        }}
                        ```
                    """
            return dummy_response
        else:
            return f"""```json
                        {{
                            "do_intervene": true,
                            "intervention_justification": "dummy justification",
                            "text": "{str(hash(prompt))}"
                        }}
                        ```
                    """
