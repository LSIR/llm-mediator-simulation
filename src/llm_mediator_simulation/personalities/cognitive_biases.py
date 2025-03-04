import csv
import os
from dataclasses import dataclass
from enum import Enum


class ReasoningError(Enum):
    """Base class for reasoning errors."""

    def __str__(self) -> str:
        """Return a printable version of the reasoning error."""
        return self.value.name.capitalize()


@dataclass
class CognitiveBiasValue:
    """Typing for the values of a cognitive bias."""

    name: str
    description: str
    group: str | None = None
    type: str | None = None

    def get_description(self) -> str:
        return self.description


def load_biases_from_csv(file_path: str = "data/cognitive-bias-tree.csv") -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}. Please provide a valid path to the CSV file to load cognitive biases."
        )

    with open(file_path, mode="r", newline="", encoding="utf-8") as csvfile:
        # remove non-breaking space characters from the CSV file such as '\xa0'
        reader = csv.DictReader(csvfile)
        enum_entries = {}
        for row in reader:
            row = {k: v.replace("\xa0", " ") for k, v in row.items()}
            enum_name = row["Name"].upper().replace(" ", "_")
            name = row["Name"]
            group = row["Group"]
            type_ = row["Type"]
            description = row["Description"]

            # Add the Enum entry dynamically
            bias_value = CognitiveBiasValue(name, description, group, type_)

            enum_entries[enum_name] = bias_value
        # Add each entry to the CognitiveBias Enum
        global CognitiveBias
        CognitiveBias = Enum("CognitiveBias", enum_entries, type=ReasoningError)
        CognitiveBias.__doc__ = """"Cognitive biases for agents.
    Based on the List of cognitive biases:
        - https://en.wikipedia.org/wiki/List_of_cognitive_biases
        - csv file of the Wikipedia page https://github.com/UN-AVT/kamino-source/blob/95c6e151e69d55d61a3dc15a3344a75a4abe2cbc/sources/14-trees/2-node-link/2-radial/archetypes/cognitive-biases/cognitive-bias-tree.csv
    """


load_biases_from_csv()
