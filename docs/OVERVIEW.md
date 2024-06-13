# OVERVIEW

This project aims at simulating online political debates using LLMs, and measuring the efficiency of LLMs as debate mediators by monitoring debate quality and toxicity.

## Configuration

Debate simulations are configured through 2 objects: `DebateConfig` and `Debater`.
These classes can be found [here](../llm_mediator_simulations/simulation/configuration.py).

### Debate Configuration

The `DebateConfig` class holds the debate metadata:

- Which statement is being discussed
- The debate prompt context (an online political debate)
- Message prompt instructions (answer with short chat messages)

### Debater Configuration

The `Debater` class holds the configuration of individual debater agents.
A debater group config consists in a list of `Debater` instances.

- The debater's position (for or against the debate statement)
- The debater personality (a list of `Personality` enum members)

## Models

Debate messages are generated using LLMs. LLM wrapper classes are provided for `ChatGPT`, `Gemini` and `Mistral` models in the [`models`](../llm_mediator_simulations/models) folder.

You need an API key in order to use any of these models.

## Metrics

The debate quality and toxicity are computed through metrics.
We defined 2 types of metrics: Perspective API toxicity score, and custom debate quality statements evaluated by LLMs as judges.

You can define which metrics to compute on-the-fly for debate messages using the [`MetricsHandler`](../llm_mediator_simulations/metrics/metrics_handler.py) class.

Perspective API metrics require a Perspective API key, whereas custom quality metrics require a LLM model wrapper.

Note that Perspective API limits queries to 1 per second. The [`PerspectiveScorer`](../llm_mediator_simulations/metrics/perspective_api.py) wrapper class includes a built in rate limiter to avoid failing requests.

## Summary

The debater memory is handled through a [`SummaryHandler`](../llm_mediator_simulations/simulation/summary_handler.py) class that handles summarizing the ongoing debate using a LLM model, and remembers the `N` latest messages in order to emphasize their importance in the prompt for the next message. You can customize how many messages should be emphasized when building this class.

## The Debate

All these classes are needed in order to create a [`Debate`](../llm_mediator_simulations/simulation/debate.py) instance that will automatically generate the debate and compute its metrics.

Once the debate is finished, you can save its configuration and messages to a pickle file for later use.
Visit the [`example_debate.py`](../examples/example_debate.py) file for an example usage.
