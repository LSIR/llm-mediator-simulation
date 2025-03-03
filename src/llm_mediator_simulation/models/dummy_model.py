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

            # The problem is the prompt returns something like this:
            # """
            # Some text to ignore
            # Answer in valid JSON format with the following structure only:
            # ```json
            # {
            #     "agreeableness": a string ("more", "less", or "same") to update this trait
            #     ...
            # }
            # ```
            # """

            # So we neet to:
            # 1. Change all the double quotes in parentheses to single quotes, e.g. ("more", "less", or "same") -> ('more', "less', or 'same)
            # 2. Wrap the text after colons in double quotes, e.g. "agreeableness": a string ('more', 'less', or 'same') to update this trait -> "agreeableness": "a string ('more', 'less', or 'same') to update this trait"

            # Old approach, not elegant:
            # prompt = re.sub(r'"(.*?)"(:)', r'"""\1"""\2', prompt)
            # prompt = re.sub(r"(: )(.*?)(,?\n)", r'\1"""\2"""\3', prompt)
            # prompt = re.sub(r'"', r"'", prompt)
            # prompt = re.sub(r"'''", r'"', prompt)

            # New approach, more elegant:
            # Starting with 1.
            # Pattern retrieves text between parenthesis: e.g. ("more", "less", or "same")
            pattern = re.compile(r"(\([^)]*\))")

            def replace_func(match: re.Match) -> str:
                # Extract the text inside the parentheses (group(1))
                inside = match.group(1)
                # Replace all double quotes in that substring with single quotes
                inside = inside.replace('"', "'")
                # Return the modified substring, putting the parentheses back
                return f"{inside}"

            # Use the pattern and the replacement function on the entire string
            prompt = pattern.sub(replace_func, string=prompt)

            # 1. done, now addressing 2.
            # Put the text after colons in double quotes
            prompt = re.sub(r"(: )(.*?)(,?\n)", r'\1"\2"\3', prompt)

            answer_format: dict[str, str] = parse_llm_json(prompt)

            str_key_update = ""
            for key, value in answer_format.items():
                # Extract the first example in parentheses, e.g. ('more', 'less', or 'same') -> 'more'
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
                            "intervention_justification": "Dummy justification",
                            "text": "Dummy text: {str(hash(prompt))}"
                        }}
                        ```
                    """
