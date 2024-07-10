"""Calls to Perspective API for toxicity anaysis."""

import time
from datetime import datetime

from perspective import PerspectiveAPI


class PerspectiveScorer:
    """Toxicity scorer using Perspective API."""

    def __init__(self, api_key: str, rate_limit: float | None = 1.1):
        """Instanciate the Perspective API scorer.

        Args:
            api_key (str): The Perspective API key.
            rate_limit (float, optional): The rate limit to avoid exceeding the quota. Defaults to 1.1 because Perspective API has a default rate limit of 1 query per second.
        """
        self.client = PerspectiveAPI(api_key=api_key)
        self.last_call: datetime | None = None
        self.rate_limit = rate_limit

    def score(self, text: str) -> float:
        """Score the toxicity of a text."""

        if self.last_call is not None and self.rate_limit is not None:
            time_since_last_call = (datetime.now() - self.last_call).total_seconds()
            if time_since_last_call < self.rate_limit:
                time_to_sleep = self.rate_limit - time_since_last_call
                time.sleep(time_to_sleep)

        score = self.client.score(text)["TOXICITY"]
        self.last_call = datetime.now()
        return score
