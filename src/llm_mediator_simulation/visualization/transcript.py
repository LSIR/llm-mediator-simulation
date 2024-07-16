"""Generate a human-readable conversation transcript from a pickled debate."""

from llm_mediator_simulation.simulation.configuration import (
    DebateConfig,
    DebatePosition,
    Debater,
)
from llm_mediator_simulation.simulation.debate import DebatePickle
from llm_mediator_simulation.utils.types import Intervention


def debate_interventions_transcript(
    interventions: list[Intervention], debaters: list[Debater]
) -> str:
    """Write a list of interventions into a text transcript"""

    lines: list[str] = []

    for intervention in interventions:
        if intervention.text is None or intervention.text == "":
            continue

        authorId = intervention.authorId
        author = (
            debaters[authorId].name
            if authorId is not None and authorId < len(debaters)
            else "Mediator"
        )

        line = f"{intervention.timestamp.strftime('%H:%M:%S')} - {author}: {intervention.text}"
        lines.append(line)

    return "\n\n".join(lines)


def debate_config_transcript(config: DebateConfig) -> str:
    """Write a debate configuration into a text transcript"""

    return config.statement


def debate_participants_transcript(debaters: list[Debater]) -> str:
    """Write a list of debaters into a text transcript"""

    lines: list[str] = []

    for debater in debaters:
        line = f"{debater.name} is arguing {'for' if debater.position == DebatePosition.FOR else 'against'} the statement."
        lines.append(line)

    return "\n".join(lines)


def debate_transcript(debate: DebatePickle) -> str:
    """Generate a full human-readable conversation transcript from a pickled debate"""

    return f"""Debate transcript
Statement: {debate_config_transcript(debate.config)}

Participants:
{debate_participants_transcript(debate.debaters)}

Transcript:
{debate_interventions_transcript(debate.messages, debate.debaters)}
"""


def save_transcript(debate: DebatePickle | str, path: str) -> None:
    """Save a human-readable conversation transcript to a file"""

    if isinstance(debate, DebatePickle):
        debate = debate_transcript(debate)

    with open(path, "w") as f:
        f.write(debate)
