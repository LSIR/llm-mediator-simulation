"""Helper module for LLM JSON answer processing."""

import json
from typing import TypeVar


def json_prompt(format: dict[str, str]) -> str:
    """Generate a prompt for a json response with the given format and descriptions.

    Args:
        format (dict[str, str]): The format of the json response, with a description for each key.
    """
    sep = "\n"

    return f"""Answer in JSON format with the following structure:
{{
    {f",{sep}    ".join(f'"{key}": {value}' for key, value in format.items())}
}}
"""


def validate_shallow_json(data: dict, typedDict) -> bool:
    """Validate that a shallow JSON object can be coerced to the given TypedDict instance.
    Excess fields are allowed.

    Args:
        data (dict): The JSON object to validate.
        typedDict (TypedDict): The TypedDict instance to which the JSON object should be coerced.

    Returns:
        bool: True if the JSON object can be coerced to the TypedDict instance, False otherwise.
    """
    for key, expected_type_hint in typedDict.__annotations__.items():
        if key not in data:
            print(f"Validation failed: Missing key '{key}'")
            return False
        if not isinstance(data[key], expected_type_hint):
            print(f"Validation failed: Key '{key}' is not of type {expected_type_hint}")
            return False
    return True


# Generic TypedDict type
T = TypeVar("T")


def parse_llm_json(llm_json: str, typedDict: type[T] | None = None) -> T:
    """Parse a LMM JSON response, and ensure that the resulting response can be coerced to the
    given TypedDict instance. Excess fields are allowed.

    Args:
        llm_json (str): The LLM JSON response.
        typedDict (TypedDict | None): The TypedDict instance to which the response should be coerced. \
If None, no validation is performed.

    Throws:
        ValueError: If the response cannot be coerced to the given TypedDict instance.
        JsonDecodeError: If the response is not valid JSON.
    """
    sanitized_json = llm_json.replace("```json", "").replace("```", "")

    data = json.loads(sanitized_json)

    if typedDict and not validate_shallow_json(data, typedDict):
        raise ValueError(
            "JSON response does not match the expected TypedDict instance."
        )

    return data
