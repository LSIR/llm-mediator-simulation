import json
import os
from dataclasses import dataclass
from enum import Enum

from llm_mediator_simulation.personalities.cognitive_biases import ReasoningError


@dataclass
class FallacyValue:
    """Typing for the values of a cognitive bias."""

    name: str
    description: str

    def get_description(self) -> str:
        return self.description


def load_fallacies_from_json(file_path: str = "data/fallacies.json") -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}. Please provide a valid path to the JSON file to load cognitive biases."
        )

    with open(file_path, mode="r", newline="", encoding="utf-8") as jsonfile:
        # remove non-breaking space characters from the JSON file such as '\xa0'
        fallacies = json.load(jsonfile)
        enum_entries = {}
        for fallacy in fallacies:
            enum_name = fallacy["name"].upper().replace(" ", "_")
            name = fallacy["name"]
            description = fallacy["definition"]

            # Add the Enum entry dynamically
            fallacy_value = FallacyValue(name, description)

            enum_entries[enum_name] = fallacy_value
        # Add each entry to the Fallacy Enum
        global Fallacy
        Fallacy = Enum("Fallacy", enum_entries, type=ReasoningError)
        Fallacy.__doc__ = """Fallacies for agents.
    Based on the List of fallacies:
        - https://en.wikipedia.org/wiki/List_of_fallacies
        - json file of the Wikipedia page https://github.com/keyofbpoe1/bingogame1/blob/17eb76dda74705052f7f19f20f1e10223e69b648/app/fallacies.json
    """


load_fallacies_from_json()
