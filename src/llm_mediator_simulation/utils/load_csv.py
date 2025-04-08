"""Helper to load data from a CSV file"""

from datetime import datetime

import polars as pd

from llm_mediator_simulation.simulation.debater.config import (
    DebaterConfig,
)
from llm_mediator_simulation.utils.types import Intervention


def load_reddit_csv_conv(
    path: str, truncated_num: int | None = 2, force_truncated_order: bool = False
) -> tuple[list[DebaterConfig], list[Intervention], list[str] | None]:
    """Extract a list of debater configs and interventions from a chat CSV file.
    Note that no personalities and debate positions can be inferred for debaters.
    Positions default to FOR.
    """

    assert (
        not (force_truncated_order) or truncated_num
    ), "If force_truncated_order is True, truncated_num must be set to a value greater than 0."

    df = pd.read_csv(path)

    if force_truncated_order:
        assert (
            truncated_num
        ), "If force_truncated_order is True, truncated_num must be set to a value greater than 0."
        forced_debater_order = df[-truncated_num:]["User Name"].to_list()
    else:
        forced_debater_order = None

    # Check for columns
    # User ID is 0 for Reddit user replying to OP (disagrees with statement)
    # User ID is 1 for OP (agrees with statement)
    expected_columns = ["User ID", "User Name", "Text", "Timestamp"]

    if not all(col in df.columns for col in expected_columns):
        raise ValueError(
            f"Expected columns: {expected_columns} to be present in a chat CSV file"
        )

    # Extract users
    users = df.select("User ID", "User Name").unique(subset=["User ID"])
    users_id_map = dict(zip(users["User ID"], users["User Name"]))

    debaters: dict[int, DebaterConfig] = {}
    # ID to debater config
    for id, name in users_id_map.items():
        if name in ["[deleted]" or ["removed"]]:
            name = ""  # We assume that all messages written by deleted or removed users are written by the same user, which might not be always true...

        debaters[id] = DebaterConfig(
            name,
        )

    if truncated_num is not None:
        # Remove the last n rows of the dataframe
        df = df[:-truncated_num]

    # Extract interventions
    app_interventions = df.select("User ID", "Text", "Timestamp")

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
            timestamp=datetime.strptime(
                datetime.fromtimestamp(app_interventions["Timestamp"][i]).strftime(
                    date_format
                ),
                date_format,
            ),
        )
        for i in range(len(app_interventions))
    ]

    return list(debaters.values()), interventions, forced_debater_order


def load_deliberate_lab_csv_chat(
    path: str,
) -> tuple[list[DebaterConfig], list[Intervention], None]:
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
        id: DebaterConfig(
            name,
            topic_opinion=None,
        )
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

    return list(debaters.values()), interventions, None
