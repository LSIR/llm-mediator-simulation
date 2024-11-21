"""Debater handler class"""

from copy import deepcopy
from datetime import datetime
from typing import override
from llm_mediator_simulation.utils.decorators import retry

from llm_mediator_simulation.utils.interfaces import Promptable
from llm_mediator_simulation.models.language_model import LanguageModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debater.config import DebaterConfig
from llm_mediator_simulation.simulation.prompt import (
    debater_intervention,
    LLM_PROBA_RESPONSE_FORMAT,
    LLM_RESPONSE_FORMAT,
    debater_personality_update,
)
from llm_mediator_simulation.utils.json import (
    json_prompt,
    parse_llm_json,
)
from llm_mediator_simulation.simulation.summary.handler import SummaryHandler
from llm_mediator_simulation.utils.types import Intervention, LLMMessage



class DebaterHandler(Promptable):
    """Debater simulation handler class"""

    def __init__(
        self,
        *,
        model: LanguageModel,
        config: DebaterConfig,
        debate_config: DebateConfig,
        summary_handler: SummaryHandler,
        relative_memory: bool = False,
        latest_messages_limit: int = 2 ,
    ) -> None:
        """Initialize the debater handler.

        Args:
            model: The language model to use.
            config: The debater configuration. The debater personality will evolve during the debate.
            debate_config: The debate configuration.
            summary_handler: The conversation summary handler.
            relative_memory: Whether the summary is relative to the debater's memory. Defaults to False.
        """
        self.model = model
        self.config = config
        self.debate_config = debate_config
        self.summary_handler = summary_handler
        self.latest_messages: list[Intervention] = []
        self.relative_memory = relative_memory 
        self.memory = ""
        self._latest_messages_limit = latest_messages_limit  

    def intervention(self, update_personality=False) -> Intervention:
        """Do a debater intervention.

        Args:
            update_personality: Whether to update the debater personality based on the last messages before intervention.
        """

        # Update the debater personality
        if update_personality:
            debater_personality_update(
                model=self.model,
                debater=self.config,
                interventions=self.summary_handler.latest_messages,
            )

        # Do the intervention
        time = datetime.now()
        
        if self.relative_memory:
            response, prompt = self.intervention_from_memory()
        else:
            response, prompt = debater_intervention(
                model=self.model,
                config=self.debate_config,
                summary=self.summary_handler,
                debater=self.config,
            )

        return Intervention(
            debater=deepcopy(self.config),  # Freeze the debater configuration
            text=response["text"],
            prompt=prompt,
            justification=response["intervention_justification"],
            timestamp=datetime.now(),
        )
        
     
    
    
    ##### SUMMARY HANDLER #####
     
     
    @retry(attempts=5, verbose=True)
    def intervention_from_memory(self, seed: int | None = None) -> tuple[LLMMessage, str]:
        """Debater intervention: decision, motivation for the intervention, and intervention content."""

        prompt = f"""{self.debate_config.to_prompt()}. {self.config.to_prompt()} {self.to_prompt()}

        Do you want to add a comment to the online debate right now?
        You should often add a comment when the previous context is empty or not in the favor of your \
        position. However, you should almost never add a comment when the previous context already \
        supports your position. Use short chat messages, no more than 3 sentences.

        {json_prompt(LLM_RESPONSE_FORMAT)}
        """

        response = self.model.sample(prompt)
        return parse_llm_json(response, LLMMessage), prompt

     
    @property
    def message_speakers_and_strings(self) -> list[str]:
        """Return the name of last messages"""
        return [
        f"You said: \"{message.text}\"" if message.debater is not None and message.debater.name == self.config.name
        else f"{message.debater.name} said: \"{message.text}\"" if message.debater is not None
        else f"Mediator: {message.text}"
        for message in self.latest_messages if message.text
        ]
    
    def add_new_message(self, message: Intervention) -> None:
        """Add a new message to the latest messages list.
        Empty messages are ignored."""

        if not message.text:
            return

        self.latest_messages = (self.latest_messages + [message])[
            -self._latest_messages_limit :
        ]
       
    def regenerate_memory(self) -> str:
        """Regenerate the summary with the latest messages."""

        self.memory = self.summarize_conversation_with_last_messages_debater_version(
            self.model, self.memory, self.message_speakers_and_strings
        )

        return self.memory
    
    
    def summarize_conversation_with_last_messages_debater_version(
        self, model: LanguageModel, previous_summary: str, latest_messages_speakers: list[str]
    ) -> str:
        """Generate a summary of the given conversation, with an emphasis on the latest messages.
        Every message is labeled with the speaker.
        Args:
            model (LanguageModel): The language model to use for generating the summary.
            previous_summary (str): The previous summary of the conversation.
            latest_messages (list[str]): The latest messages in the conversation.
            speakers (list[str]): The speakers corresponding to each message in latest_messages.
        """

        separator = "\n\n"
        labeled_messages = [message for message in latest_messages_speakers]
        prompt = f"""Conversation summary: {previous_summary}

        Latest messages:
        {separator.join(labeled_messages)}

        In 3-4 sentences, summarize the conversation above from your perspective as {self.config.name}, focusing on the recent messages. Frame it according to your recollection of events, highlighting your own interpretation and understanding. When mentioning the participants, refer to them by name, emphasizing your biased viewpoint on who said what.
        """
        return model.generate_response(prompt)

    
    @override
    def to_prompt(self) -> str:
        msg_sep = "\n\n"

        if len(self.latest_messages) == 0:
            return f"""The conversation between you and {', '.join({deb.name for deb in self.summary_handler.debaters if deb.name != self.config.name})} has just started, and there are no prior messages or exchanges. Please present your initial argument on the topic"""
        return f"""Here is a recollection of the previous exchanges from your memory as {self.config.name} : 
        ""{self.memory}""
        Here are the most recent exchanged messages (you should focus your argumentation on them):
        {msg_sep.join(self.message_speakers_and_strings)}
        """
    
    
    
    ##### Personality #####

    def snapshot_personality(self) -> DebaterConfig:
        """Snapshot the current debater personality."""
        return deepcopy(self.config)
