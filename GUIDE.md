# Project Guide

This guide aims at clarifying the goal of this project, and how it is implemented.

The idea: create a framework to simulate debates (political or not) over hot topics between multiple automated debaters,
with automated mediators.

The debates are made over a statement _(ex: "We should use nuclear power")_.
Debaters are configured through a list of personality axis using positions between two opposite qualities _(ex: calm vs toxic)_.
Their personnalities can evolve over time, the goal being to reduce toxicity.
Given a debate statement, the debaters can be configured to advocate either for or against the statement.

During a debate round, one after the other, debaters make an intervention.
Inbetween debater interventions, mediators can also intervene.
Interventions are not mandatory: debaters and mediators can choose not to intervene. The motivation behind their choice is still stored in the debate logs.

Various toxicity and debate quality metrics can be analyzed over the interventions in order to fine tune the mediator prompts.

The project is structured like so:

```bash
├── examples                  # Example scripts
│
├── src/llm_mediator_simulations  # Main package
│   │
│   ├── metrics        # Metrics computations
│   ├── models         # LLM model wrappers
│   ├── simulation     # Debate simulation handler classes
│   ├── utils          # Utilities
│   └── visualization  # Visualization tools
```

See the following sections for a detailed rundown.

## Simulation

The simulation logic is separated into 4 modules in the [`simulation`](./src/llm_mediator_simulation/simulation) directory:

- debater
- mediator
- summary
- debate

Each of them has 3 components:

- `config.py`: configuration dataclass
- `handler.py`: handler logic
- `async_handler.py`: asynchronous handler logic

The configuration is isolated into dataclasses in order to easily export configurations via pickles.
Async handlers enable running multiple debates in parallel.

### Debater

Defined in the [`simulation/debater`](./src/llm_mediator_simulation/simulation/debater) directory.

A debater is configured with a name, a position (_for_ or _against_ the debate statement) and a list of personality axis.
The personality axis represent a position between two extremes, with 5 possible values: `0`, `1`, `2`, `3`, `4`.
The values on the personality axis will change over time, by `+-1` before every intervention of the targeted debater except the first.

```python
from llm_mediator_simulation.simulation.debater.config import (
    AxisPosition,
    DebatePosition,
    DebaterConfig,
    PersonalityAxis,
)

DebaterConfig(
    name="Alice",
    position=DebatePosition.AGAINST,
    personalities={
        PersonalityAxis.CIVILITY: AxisPosition.VERY_RIGHT,  # Very toxic
        PersonalityAxis.POLITENESS: AxisPosition.VERY_RIGHT,  # Very rude
        PersonalityAxis.EMOTIONAL_STATE: AxisPosition.VERY_RIGHT,  # Very angry
        PersonalityAxis.POLITICAL_ORIENTATION: AxisPosition.VERY_LEFT,  # Very conservative
    },
),
```

You do not need to manually create a `DebaterHandler`.

### Mediator

Defined in the [`simulation/mediator`](./src/llm_mediator_simulation/simulation/mediator) directory.

A mediator is configured with a "preprompt" (an instruction prompt that is inserted in the LLM prompt for mediator instructions) and a probability mapping config.
The `MediatorConfig` dataclass already has a detailed default preprompt.

The `ProbabilityMappingConfig` is an optional piece of configuration that enables forcing the mediator to intervene with a predefined mean intervention rate and standard deviation. The probability mapping is smart and takes into account the degree of willingness to intervene of the mediator at each intervention opportunity.

```python
from llm_mediator_simulation.simulation.mediator.config import MediatorConfig

mediator_config = MediatorConfig()
```

You do not need to manually create a `MediatorHandler`.

### Summary

Defined in the [`simulation/summary`](./src/llm_mediator_simulation/simulation/summary) directory.

In order to make the debaters as realistic as possible, a summary of the current debate interventions is computed and inserted into their prompt before every intervention. It also includes the last `N` interventions in full in order to lay an emphasis on the latest chat messages.

The mediator prompts will only include the latest `N` messages and not the summaries, as in real-life situations, a responsive LLM mediator setup will not have the time to recompute a real-time summary of ongoing chat exchanges before intervening.

The summary config also includes a list of the debaters in order to include the names of every debater before the summary in LLM prompts.

```python
from llm_mediator_simulation.simulation.summary.config import SummaryConfig

summary_config = SummaryConfig(latest_messages_limit=3, debaters=debaters)
```

You do not need to manually create a `SummaryHandler`.

### Debate

Defined in the [`simulation/debate`](./src/llm_mediator_simulation/simulation/debate) directory.

The debate configuration only includes the debated statement, a short situation context prompt, and prompts to include for debaters who are for/against the statement.
You may only need to modify the debated statement, as the defaults for the other fields are good enough.

```python
from llm_mediator_simulation.simulation.debate.config import DebateConfig

debate_config = DebateConfig(
    statement="We should use nuclear power.",
)
```

Once all of this configuration is defined, you are almost ready to begin simulating using the `DebateHandler` or `AsyncDebateHandler` classes!
But first, there are a few more configuration options that you may want to take into account.

## Metrics

Defined in the [`metrics`](./src/llm_mediator_simulation/metrics) directory.

In order to analyze how well your debate mediators are performing, you can define metrics to be computed on the fly after every debater intervention.

Two types or metrics can be configured:

- Perspective API (you need a perspective API key)
- LLMs as judges (a set of debate quality metrics computed via LLM calls)

Synchronous debate example:

```python
from llm_mediator_simulation.metrics.perspective_api import PerspectiveScorer
from llm_mediator_simulation.metrics.metrics_handler import MetricsHandler

metrics = MetricsHandler(
    model=mediator_model,
    argument_qualities=[
        ArgumentQuality.APPROPRIATENESS,
        ArgumentQuality.CLARITY,
        ArgumentQuality.LOCAL_ACCEPTABILITY,
        ArgumentQuality.EMOTIONAL_APPEAL,
    ],
)   perspective=PerspectiveScorer(api_key=your_perspective_api_key))
```

Use `AsyncMetricsHandler` in an asynchronous debate setting instead.

## Models

Defined in the [`models`](./src/llm_mediator_simulation/models) directory.

Different LLMs can be used for simulations.

4 model wrappers are defined:

- Gemini models (suitable for mediator simulations).
- GPT models (suitable for mediator simulations).
- Mistral models (remote and local, suitable for both).

Remember to use a model with no toxicity restrictions for debater simulations!
(Note: removing restrictions on Gemini models is not easy because the API is always changing. For this reason, we used Mistral local models).

Of course, you will need your own API keys.

Synchronous debate example:

```python
from llm_mediator_simulation.models.google_models import GoogleModel
from llm_mediator_simulation.models.gpt_models import GPTModel
from llm_mediator_simulation.models.mistral_models import MistralModel

mediator_model = GPTModel(api_key=gpt_key, model_name="gpt-4o")
mediator_model = GoogleModel(api_key=google_key, model_name="gemini-1.5-pro")
debater_model = MistralModel(api_key=mistral_key, model_name="mistral-large-latest")
```

Use the `async` model wrapper versions in an async setting instead.
