"""Helper to load data from a CSV file"""

from datetime import datetime

import polars as pd

from llm_mediator_simulation.simulation.debater.config import (
    TopicOpinion,
    DebaterConfig,
)
from llm_mediator_simulation.utils.types import Intervention


def load_csv_chat(path: str) -> tuple[list[DebaterConfig], list[Intervention]]:
    """Extract a list of debater configs and interventions from a chat CSV file.
    Note that no personalities and debate positions can be inferred for debaters.
    Positions default to FOR.
    """

    df = pd.read_csv(path)

    # Check for columns
    expected_columns = ["User ID", "User Name", "Message Type", "Text", "Timestamp"]

    if not all(col in df.columns for col in expected_columns):
        raise ValueError(
            f"Expected columns: {expected_columns} to be present in a chat CSV file"
        )

    # Extract users
    users = df.select("User ID", "User Name").unique(subset=["User ID"])
    users_id_map = dict(zip(users["User ID"], users["User Name"]))
    users_id_map[""] = "Mediator"

    # ID to debater config
    debaters: dict[str, DebaterConfig] = {
        id: DebaterConfig(name, topic_opinion=None, personalities={}) # TODO: Check if topic_opinion needed
        for id, name in users_id_map.items()
        if name != "Mediator"
    }

    # Extract interventions
    app_interventions = df.select("Message Type", "User ID", "Text", "Timestamp")

    date_format = "%a %b %d %Y (%H:%M)"
    interventions = [
        Intervention(
            debater=(
                debaters[app_interventions["User ID"][i]]
                if app_interventions["User ID"][i] != ""
                else None
            ),
            text=app_interventions["Text"][i],
            prompt="",
            justification="",
            timestamp=datetime.strptime(app_interventions["Timestamp"][i], date_format),
        )
        for i in range(len(app_interventions))
    ]

    return list(debaters.values()), interventions
