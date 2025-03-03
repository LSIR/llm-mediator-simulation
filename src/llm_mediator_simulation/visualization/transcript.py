"""Generate a human-readable conversation transcript from a pickled debate."""

from llm_mediator_simulation.personalities.scales import Likert7AgreementLevel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debate.handler import DebatePickle
from llm_mediator_simulation.simulation.debater.config import (
    DebaterConfig,
)
from llm_mediator_simulation.utils.types import Intervention


def debate_interventions_transcript(interventions: list[Intervention]) -> str:
    """Write a list of interventions into a text transcript"""

    lines: list[str] = []

    for intervention in interventions:
        if intervention.text is None or intervention.text == "":
            continue

        author = intervention.debater.name if intervention.debater else "Mediator"

        line = f"{intervention.timestamp.strftime('%H:%M:%S')} - {author}: {intervention.text}"
        lines.append(line)

    return "\n\n".join(lines)


def debate_config_transcript(config: DebateConfig) -> str:
    """Write a debate configuration into a text transcript"""

    return config.statement


def debate_participants_transcript(debaters: list[DebaterConfig]) -> str:
    """Write a list of debaters into a text transcript"""

    lines: list[str] = []

    for debater in debaters:
        if debater.topic_opinion is None:
            line = f"{debater.name} has no initial opinion on the statement."

        else:
            if debater.topic_opinion.agreement == Likert7AgreementLevel.NEUTRAL:
                line = (
                    f"{debater.name} neither agrees nor disagrees with the statement."
                )
            else:
                line = f"{debater.name} {debater.topic_opinion.agreement.value}s with the statement."
        lines.append(line)

    return "\n".join(lines)


def debate_transcript(debate: DebatePickle) -> str:
    """Generate a full human-readable conversation transcript from a pickled debate"""

    return f"""Debate transcript
Statement: {debate_config_transcript(debate.config)}

Participants:
{debate_participants_transcript(debate.debaters)}

Transcript:
{debate_interventions_transcript(debate.interventions)}
"""


def save_transcript(debate: DebatePickle | str, path: str) -> None:
    """Save a human-readable conversation transcript to a file"""

    if isinstance(debate, DebatePickle):
        debate = debate_transcript(debate)

    with open(path, "w") as f:
        f.write(debate)
