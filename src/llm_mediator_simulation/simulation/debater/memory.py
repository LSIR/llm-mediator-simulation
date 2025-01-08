import getpass
import os


#os.environ["MISTRAL_API_KEY"] = getpass.getpass()

from langchain_mistralai import ChatMistralAI
from langchain_core.documents import Document
import operator
from typing import List, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)
from langchain_core.documents import Document


def summarize_conversation_with_last_messages_debater_version(
        name:str, pre_prompt: str, model: LanguageModel, previous_memory: str, latest_messages_speakers: List[str]
    ) -> str:
    llm = ChatMistralAI(model="mistral-large-latest")
    
     # Prepare content as a list of mappings
    content = [{"context": c} for c in ([previous_memory] + latest_messages_speakers)]

    # Initial memory
    summarize_prompt = ChatPromptTemplate([
        ("human", f"{pre_prompt}. In very short, update your summarised understanding of the conversation above from your perspective as {name}, completing your previous understanding with the new messages. Frame it according to your recollection of events, highlighting your own interpretation and understanding. : {{context}}")
    ])

    initial_memory_chain = summarize_prompt | llm | StrOutputParser()

    # Refining the memory with new context
    refine_template = """
    Produce a final memory.

    Existing memory up to this point:
    {existing_answer}

    New context:
    ------------
    {context}
    ------------

    Given the new context, refine the original memory.
    """
    refine_prompt = ChatPromptTemplate([("human", refine_template)])

    refine_memory_chain = refine_prompt | llm | StrOutputParser()

    # Define state graph functions
    def generate_initial_memory(state: "State", config: RunnableConfig):
        memory = initial_memory_chain.invoke(
            state["contents"][0],
            config,
        )
        return {"memory": memory, "index": 1}

    def refine_memory(state: "State", config: RunnableConfig):
        content = state["contents"][state["index"]]
        memory = refine_memory_chain.invoke(
            {"existing_answer": state["memory"], "context": content},
            config,
        )
        return {"memory": memory, "index": state["index"] + 1}

    def should_refine(state: "State") -> Literal["refine_memory", END]:
        if state["index"] >= len(state["contents"]):
            return END
        else:
            return "refine_memory"

    # Set up the state graph
    graph = StateGraph(State)
    graph.add_node("generate_initial_memory", generate_initial_memory)
    graph.add_node("refine_memory", refine_memory)

    graph.add_edge(START, "generate_initial_memory")
    graph.add_conditional_edges("generate_initial_memory", should_refine)
    graph.add_conditional_edges("refine_memory", should_refine)

    app = graph.compile()

    # Execute the graph
    state = {"contents": content, "index": 0, "memory": ""}
    for step in app.stream(state, stream_mode="values"):
        if memory := step.get("memory"):
            final = memory

    return final


def async_summarize_conversation_with_last_messages_debater_version(
        self, model: LanguageModel, previous_memory: str, latest_messages_speakers: list[str]
    ) -> str:
    llm = ChatMistralAI(model="mistral-large-latest")
    content = [previous_memory] + latest_messages_speakers


    # Initial memory
    summarize_prompt = ChatPromptTemplate(
        [
            ("human", "Write a concise memory of the following: {context}"),
        ]
    )

    initial_memory_chain = summarize_prompt | llm | StrOutputParser()

    # Refining the memory with new docs
    refine_template = """
    Produce a final memory.

    Existing memory up to this point:
    {existing_answer}

    New context:
    ------------
    {context}
    ------------

    Given the new context, refine the original memory.
    """
    refine_prompt = ChatPromptTemplate([("human", refine_template)])

    refine_memory_chain = refine_prompt | llm | StrOutputParser()


    
    # We define functions for each node, including a node that generates
    # the initial memory:
    async def generate_initial_memory(state: State, config: RunnableConfig):
        memory = await initial_memory_chain.ainvoke(
            state["contents"][0],
            config,
        )
        return {"memory": memory, "index": 1}


    # And a node that refines the memory based on the next content
    async def refine_memory(state: State, config: RunnableConfig):
        content = state["contents"][state["index"]]
        memory = await refine_memory_chain.ainvoke(
            {"existing_answer": state["memory"], "context": content},
            config,
        )

        return {"memory": memory, "index": state["index"] + 1}


    # Here we implement logic to either exit the application or refine
    # the memory.
    def should_refine(state: State) -> Literal["refine_memory", END]:
        if state["index"] >= len(state["contents"]):
            return END
        else:
            return "refine_memory"


    graph = StateGraph(State)
    graph.add_node("generate_initial_memory", generate_initial_memory)
    graph.add_node("refine_memory", refine_memory)

    graph.add_edge(START, "generate_initial_memory")
    graph.add_conditional_edges("generate_initial_memory", should_refine)
    graph.add_conditional_edges("refine_memory", should_refine)
    app = graph.compile()

    final = None
    for step in app.astream(
        {"contents": content},
        stream_mode="values",
    ):
        if memory := step.get("memory"):
            
            
            final = memory
    return final








# We will define the state of the graph to hold the document
# contents and memory. We also include an index to keep track
# of our position in the sequence of memory.
class State(TypedDict):
    contents: List[str]
    index: int
    memory: str


    