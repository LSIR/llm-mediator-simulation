"""Format lists for prompts."""

import difflib


def format_list(str_list: list[str]) -> str:
    """Format a list of strings into a sentence."""
    if len(str_list) > 2:
        return f"{', '.join(str_list[:-1])}, and {str_list[-1]}"
    elif len(str_list) == 2:
        return f"{str_list[0]} and {str_list[1]}"
    elif len(str_list) == 1:
        return f"{str_list[0]}"
    else:
        raise ValueError("Empty list")


def format_list_and_conjugate_be(str_list: list[str]) -> str:
    return f"{format_list(str_list)} {conjugate_be(str_list)}"


def conjugate_be(str_list: list[str]) -> str:
    if len(str_list) > 1:
        return "are"
    else:  # len(str_list) == 1
        return "is"


def show_diffs(str1: str, str2: str):
    for i, s in enumerate(difflib.ndiff(str1, str2)):
        if s[0] == " ":
            continue
        elif s[0] == "-":
            print('Delete "{}" from position {}'.format(s[-1], i))
        elif s[0] == "+":
            print('Add "{}" to position {}'.format(s[-1], i))
