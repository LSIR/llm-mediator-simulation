"""Debater handler class"""
import json
import random
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
        memory_model : LanguageModel,
        config: DebaterConfig,
        debate_config: DebateConfig,
        summary_handler: SummaryHandler,
        relative_memory: bool = False,
        latest_messages_limit: int = 4 ,
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
        self.memory_model = memory_model
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
        
     
    
    def fallacy_prompt(self):
        """
        Modify the prompt to make the Debater sensible to bias and fallacious reasoning
        with a certain probability.
        """
        
        # Load fallacies from the JSON file
        with open('src/llm_mediator_simulation/utils/fallacies.json', 'r') as file:
            fallacies = json.load(file)

        # Randomly select a fallacy
        selected_fallacy = random.choice(fallacies)

        # Define the probability of detecting a fallacy
        probability_of_detection = 0.8
        prompt =" "
        if random.random() < probability_of_detection:
            fallacy_name = selected_fallacy['name']
            fallacy_definition = selected_fallacy['definition']
            prompt = f"You are sensitive to this fallacy: {fallacy_name}. {fallacy_definition}."
        return prompt
        
        
    def cognitive_biais_prompt(self):
        """
        Modify the prompt to make the Debater sensible to bias and fallacious reasoning
        with a certain probability.
        """
            
        # Load fallacies from the JSON file
        with open('src/llm_mediator_simulation/utils/cognitive_bias.json', 'r') as file:
            cognitive_biais = json.load(file)

        # Randomly select a cognitive_biais
        selected_cognitive_biais = random.choice(cognitive_biais)

        # Define the probability of detecting a cognitive_biais
        probability_of_detection = 0.8
        prompt =" "
        if random.random() < probability_of_detection:
            cognitive_biais_name = selected_cognitive_biais['name']
            cognitive_biais_definition = selected_cognitive_biais['definition']
            prompt = f"You are sensitive to this cognitive biais: {cognitive_biais_name}. {cognitive_biais_definition}."
        return prompt
    
    
    
    def intervention_from_memory(self, seed: int | None = None) -> tuple[LLMMessage, str]:
        """Debater intervention using two LLMs: Thought Crafter and Speaker."""
        
        # Step 1: Thought Crafter Prompt
        thought_crafter_prompt = f"""
        {self.debate_config.to_prompt()} {self.config.to_prompt()}. 
        {self.to_prompt()}
        
        Analyze the context {f"and prior exchanges in the online discussion to identify gaps or unaddressed angles in the conversation.Be aware that some of these previous messages may be sarcastic :"if len(self.latest_messages) > 0 else "."}
        Craft a concise yet simple argument idea based on the persona's beliefs and the debate's rules. Make sure you defend your position assertively.
        {f"Avoid redundancy by ensuring the idea adds new perspectives or counters previous arguments effectively.\
        If needed, make sure you get triggered, angry, and toxic—let the conversation get heated and intense! Call out any contradictions and challenge every weak point in the discussion!" if len(self.latest_messages) > 0 else ""}
        {self.fallacy_prompt()}
        {self.cognitive_biais_prompt()}
        
        Generate an idea for the response in JSON format:
        ```json
        {{
            "core_argument": "The main point or counterpoint Bob wants to introduce.",
            "supporting_evidence": "Any reasoning, analogy, or moral principle that backs the argument.",
            "anticipated_opponent_reaction": "A likely counterargument or reaction from Alice.",
            "response_strategy": "How Bob might refine his argument to address anticipated reactions."
        }}
        ```
        """
     
        
        # Get the Thought Crafter's response
        thought_crafter_response = self.memory_model.sample(thought_crafter_prompt)
        thought_crafter_data = parse_llm_json(thought_crafter_response)

        # Step 2: Speaker Prompt
        speaker_prompt = f"""
            {self.debate_config.to_prompt()}.
            You are {self.config.name}, {self.config.to_prompt_for_against()}
            {f"tasked with formulating a conversational response in line with the last speaker's persona." if len(self.latest_messages) > 0 else "tasked with formulating an opinion to start a conversation."}
            {f"""
            Use the provided argument blueprint to construct your response to answer to this message : {self.latest_messages[-1]}.
            Ensure the tone is ironic, sarcastic, and conversational.
            Use sarcasm and exaggeration to point out the absurdity of your opponent's opinion. For example: 'Love when politicians fly private jets to climate change conferences to tell us to carpool.', 
            'Nothing says ‘freedom’ like the government telling me what books I can read.', 'Nothing says 'pro-life' like cutting funding for programs that support children and families.'
            You can ask questions, insult or be mean if you want. 

            Follow these arguments:
            
            Core Argument: {thought_crafter_data.get("core_argument", "")}
            Supporting Evidence: {thought_crafter_data.get("supporting_evidence", "")}
            Anticipated Opponent Reaction: {thought_crafter_data.get("anticipated_opponent_reaction", "")}
            Response Strategy: {thought_crafter_data.get("response_strategy", "")}
            .""" if len(self.latest_messages) > 0 else "Assert your position."} \
            Keep it concise (maximum 3 sentences). \
            {json_prompt(LLM_RESPONSE_FORMAT)}
            """

        # Get the Speaker's response
        speaker_response = self.model.sample(speaker_prompt)
        
        speaker_response = speaker_response.replace("Alice,", "").replace("Bob,", "")

        return parse_llm_json(speaker_response, LLMMessage), speaker_prompt



     
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
        model = self.memory_model, previous_memory=self.memory, latest_messages_speakers=self.message_speakers_and_strings
        )

        return self.memory
    
    
    def summarize_conversation_with_last_messages_debater_version(
        self, model: LanguageModel, previous_memory: str, latest_messages_speakers: list[str]
    ) -> str:
        """Generate a summary of the given conversation, with an emphasis on the latest messages.
        Every message is labeled with the speaker.
        Args:
            model (LanguageModel): The language model to use for generating the summary.
            previous_memory (str): The previous summary of the conversation.
            latest_messages (list[str]): The latest messages in the conversation.
            speakers (list[str]): The speakers corresponding to each message in latest_messages.
        """

        separator = "\n\n"
        labeled_messages = [message for message in latest_messages_speakers]
        
        
        format = "{'memory': 'your understanding of the conversation' }"
        if previous_memory == "":
            prompt = f""" {self.debate_config.to_prompt()}. {self.config.to_prompt()} \
            Summarize these messages: labeled_messages from your biased understanding of the conversation, from your perspective as {self.config.name}: \
            {separator.join(labeled_messages)}
            """
        
        else :
        
            prompt = f"""
            {self.debate_config.to_prompt()}. {self.config.to_prompt()}
            Here is your previous understanding of the conversation: {previous_memory}
            
            These are new messages:
            {separator.join(labeled_messages)}

            In short, update your summarised understanding of the conversation above from your perspective as {self.config.name}, completing your previous understanding with the new messages. Frame it according to your recollection of events, highlighting your own interpretation and understanding. This updated summary needs to be in 50 words.
            """
        #Generate the response as a JSON object with the following structure:
        response = model.generate_response(prompt=prompt)

        return response
    

    
    @override
    def to_prompt(self) -> str:
        msg_sep = "\n\n"

        if len(self.latest_messages) == 0:
            return f"""Your conversation with {', '.join({deb.name for deb in self.summary_handler.debaters if deb.name != self.config.name})} has just started, and there are no prior messages or exchanges. Please present your initial statement about the topic without adressing any other person."""
        return f"""Here is a recollection of the previous exchanges from your memory as {self.config.name} : 
        "{self.memory}"
        Here are the most recent exchanged messages :
        {msg_sep.join(self.message_speakers_and_strings)}
        """
    
    
    
    ##### Personality #####

    def snapshot_personality(self) -> DebaterConfig:
        """Snapshot the current debater personality."""
        return deepcopy(self.config)
