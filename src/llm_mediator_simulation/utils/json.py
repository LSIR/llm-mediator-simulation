"""Helper module for LLM JSON answer processing."""

import json
import re
from typing import TypeVar


def json_prompt(format: dict[str, str]) -> str:
    """Generate a prompt for a json response with the given format and descriptions.

    Args:
        format (dict[str, str]): The format of the json response, with a description for each key.
    """
    sep = "\n"

    return f"""Answer in valid JSON format with the following structure only:
```json
{{
    {f",{sep}    ".join(f'"{key}": {value}' for key, value in format.items())}
}}
```
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


def extract_json(string: str) -> str:
    """Extract the last JSON code block from a string. It is delimited by triple backticks."""

    start = string.rfind("```json")
    end = string.rfind("```")

    if start == -1 or end == -1 or start >= end:
        start_match = list(re.finditer(r"\{\n?", string))
        start = start_match[-1].start() if start_match else -1
        end_match = re.search(r"\n?\s*}", string)
        if end_match is None:
            raise ValueError("No JSON code block found.")
        end = end_match.end()
        if start == -1 or end == -1 or start >= end:
            raise ValueError("No JSON code block found.")

        return string[start:end]

    return string[start + len("```json") : end]


def parse_llm_json(llm_json: str, typedDict: type[T] | None = None) -> T:
    """Parse a LLM JSON response, and ensure that the resulting response can be coerced to the
    given TypedDict instance. Excess fields are allowed.

    Args:
        llm_json (str): The LLM JSON response.
        typedDict (TypedDict | None): The TypedDict instance to which the response should be coerced. \
If None, no validation is performed.

    Throws:
        ValueError: If the response cannot be coerced to the given TypedDict instance.
        JsonDecodeError: If the response is not valid JSON.
    """
    sanitized_json = extract_json(llm_json)

    data = json.loads(sanitized_json)
    # With OlMo2, sometimes the JSON key "intervention_justification" is spelled "intervention_justifyation"...
    # So we traet any key containing "intervention" as "intervention_justification"
    keys = list(data.keys())
    for key in keys:
        if "intervention" in key and key != "intervention_justification":
            data["intervention_justification"] = data.pop(key)

    # If the JSON is not valid, raise an error
    if typedDict and not validate_shallow_json(data, typedDict):
        raise ValueError(
            "JSON response does not match the expected TypedDict instance."
        )

    return data


def parse_llm_jsons(
    llm_jsons: list[str], typedDict: type[T] | None = None
) -> tuple[list[T], list[int]]:
    """Parse a list of LLM JSON responses, and ensure that the resulting responses can be coerced to the
    given TypedDict instance. Excess fields are allowed.

    Args:
        llm_jsons (list[str]): The list of LLM JSON responses.
        typedDict (TypedDict | None): The TypedDict instance to which the responses should be coerced. \
If None, no validation is performed.
    """

    coerced: list[T] = []
    failed: list[int] = []

    for i, llm_json in enumerate(llm_jsons):
        try:
            coerced.append(parse_llm_json(llm_json, typedDict))
        except ValueError:
            failed.append(i)

    return coerced, failed
