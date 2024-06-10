"""Calls to Perspective API for toxicity anaysis."""

from perspective import PerspectiveAPI


class PerspectiveScorer:
    """Toxicity scorer using Perspective API."""

    def __init__(self, api_key: str):
        self.client = PerspectiveAPI(api_key=api_key)

    def score(self, text: str) -> float:
        """Score the toxicity of a text."""
        # TODO: see all options, test them... (untested as the API is not activated for this project yet)
        result = self.client.score(text)
        print(result)
        return result["TOXICITY"]
