# Project Guide

This guide aims at clarifying the goal of this project, and how it is implemented.

The idea: create a framework to simulate debates (political or not) over controversial topics between multiple automated debaters,
with automated mediators.

The debates are made over a statement _(ex: "We should use nuclear power")_.
Debaters are configured through an instance of ```Personality``` with optional and configurable demographics, traits, facets, moral foundations, basic human values, cognitive biases, fallacies, ideologies, vote history, and other opinions and beliefs.
The personality can evolve over time.
Given a debate statement, the debaters can be configured to advocate on a 7-point Likert scale ranging from ```STRONGLY_DISAGREE``` to ```STRONGLY_AGREE``` with the statement.

During a debate round, one after the other, debaters make an intervention.
Inbetween debater interventions, mediators can also intervene.
Interventions are not mandatory: debaters and mediators can choose not to intervene. The motivation behind their choice is still stored in the debate logs.

Various toxicity and debate quality metrics can be analyzed over the interventions in order to fine tune the mediator prompts.

An experimental Streamlit app is provided to run and visualize debates in real-time thourgh a web user interface.

The project is structured like so:

```bash
├── app                       # Streamlit Web app
│
├── examples                  # Example scripts
│
├── src/llm_mediator_simulations  # Main package
│   │
│   ├── configs        # Hydra configuration files
│   ├── metrics        # Metrics computations
│   ├── models         # LLM model wrappers
│   ├── personalities  # Debater personalities
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

A debater is configured with a name, a opinion on the current debate's topic (on a 7-point Likert scale ranging from `STRONGLY_DISAGREE` to `STRONGLY_AGREE` with the statement) and a personality.

The personality is configured with optional and configurable fields such as demographics, traits, facets, moral foundations, basic human values, cognitive biases, fallacies, ideologies, vote history, and other opinions and beliefs. Each personality field is an optional collection, that can be configured as follows:

- `demographic_profile`: a dictionary of `DemographicCharacteristic` enum members (e.g. `ETHNICITY`, `AGE`, etc.) with a free-form `str` value.
- `traits`: a dictionary of `Trait` enum members (e.g. `OPENNESS`, `CONSCIENTIOUSNESS`, etc.) with a 3-point Likert scale value ranging from `LOW` to `HIGH`. Traits can also be a list of `Trait` enum members handled as if they were all set to `HIGH`.
- `facets`: a dictionary of `Facet` enum members (e.g. `ANXIETY`, `ANGER`, etc.) with a binary value `POSITIVE` or `NEGATIVE`. Facets can also be a list of `Facet` enum members handled as if they were all set to `POSITIVE`.
- `moral_foundations`: a dictionary of `MoralFoundation` enum members (e.g. `CARE_HARM`, `FAIRNESS_CHEATING_EQUALITY`, etc.) with a 5-point Likert scale value ranging from `NOT_AT_ALL` to `EXTREMELY`. Moral foundations can also be a list of `MoralFoundation` enum members.
- `basic_human_values`: a dictionary of `BasicHumanValue` enum members (e.g. `SELF_DIRECTION`, `STIMULATION`, etc.) with a 5-point Likert scale value ranging from `NOT_AT_ALL` to `EXTREMELY`. Basic human values can also be a list of `BasicHumanValue` enum members handled as if they were all set to `IMPORTANT`.
- `cognitive_biases`: a list of `CognitiveBias` enum members (e.g. `ANCHORING`, `CONFIRMATION_BIAS`, etc.).
- `fallacies`: a list of `Fallacy` enum members (e.g. `AD_HOMINEM`, `APPEAL_TO_AUTHORITY`, etc.).
- `vote_last_presidential_election`: a free-form `str`
- `ideologies`: a list of `Ideology` enum members (e.g. `ECONOMIC`, `SOCIAL`, etc.) with a 7(+2)-point likert scale value ranging from `EXTREMELY_LIBERAL` to `EXTREMELY_CONSERVATIVE` (+ `LIBERTARIAN` and `INDEPENDENT`)
- `agreement_with_statements`: a dict of `str` with a 7-point Likert scale value ranging from `STRONGLY_DISAGREE` to `STRONGLY_AGREE`.
- `likelihood_of_beliefs`: a dict of `str` with a 11-point Likert scale value ranging from `CERTAINLY_FALSE` to `CERTAINLY_TRUE`.
- `free_form_opinions`: a list of free-form `str`.



The personality fields (except `demographic_profile`, `vote_last_presidential_election`, and `free_form_opinions`) can optionally change over time by setting the `variable_[field_name]` to `True` (`False` by default). 

Variable cognitive biases and fallacies are randomly drawn from the comprehensive Wikipedia lists of cognitive biases and fallacies.

For other vaiable fields, the agent will be prompted to choose to update the field's Likert-scale value with `"more"`, `"less"`, or `"same"` (and `"yes"` or `"no"` for variable facets) before every intervention of the targeted debater (except before the first intervention of the debate).

```python
from llm_mediator_simulation.personalities.cognitive_biases import CognitiveBias
from llm_mediator_simulation.personalities.demographics import DemographicCharacteristic
from llm_mediator_simulation.personalities.facets import PersonalityFacet
from llm_mediator_simulation.personalities.fallacies import Fallacy
from llm_mediator_simulation.personalities.human_values import BasicHumanValues
from llm_mediator_simulation.personalities.ideologies import Ideology, Issues
from llm_mediator_simulation.personalities.moral_foundations import MoralFoundation
from llm_mediator_simulation.personalities.personality import Personality
from llm_mediator_simulation.personalities.scales import (
    KeyingDirection,
    Likert3Level,
    Likert5ImportanceLevel,
    Likert5Level,
    Likert7AgreementLevel,
    Likert11LikelihoodLevel,
)
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.simulation.debater.config import (
    DebaterConfig,
    TopicOpinion
)
    

demo_profile = {
    DemographicCharacteristic.ETHNICITY: "White",
    DemographicCharacteristic.BIOLOGICAL_SEX: "male",
}

traits = {
    PersonalityTrait.AGREEABLENESS: Likert3Level.HIGH,
    PersonalityTrait.CONSCIENTIOUSNESS: Likert3Level.LOW,
    PersonalityTrait.EXTRAVERSION: Likert3Level.AVERAGE,
    PersonalityTrait.NEUROTICISM: Likert3Level.HIGH,
    PersonalityTrait.OPENNESS: Likert3Level.LOW,
}

facets = {
    PersonalityFacet.ALTRUISM: KeyingDirection.POSITIVE,
    PersonalityFacet.ANGER: KeyingDirection.NEGATIVE,
    PersonalityFacet.ANXIETY: KeyingDirection.POSITIVE,
}

moral_foundations = {
    MoralFoundation.CARE_HARM: Likert5Level.EXTREMELY,
    MoralFoundation.AUTHORITY_SUBVERSION: Likert5Level.SLIGHTLY,
    MoralFoundation.FAIRNESS_CHEATING_PROPORTIONALITY: Likert5Level.MODERATELY,
    MoralFoundation.LOYALTY_BETRAYAL: Likert5Level.NOT_AT_ALL,
    MoralFoundation.SANCTITY_DEGRADATION_PURITY: Likert5Level.SLIGHTLY,
}

basic_human_values = {
    BasicHumanValues.SELF_DIRECTION_THOUGHT: Likert5ImportanceLevel.IMPORTANT,
    BasicHumanValues.STIMULATION: Likert5ImportanceLevel.VERY_IMPORTANT,
    BasicHumanValues.HEDONISM: Likert5ImportanceLevel.VERY_IMPORTANT,
    BasicHumanValues.ACHIEVEMENT: Likert5ImportanceLevel.IMPORTANT,
    BasicHumanValues.POWER_DOMINANCE: Likert5ImportanceLevel.NOT_IMPORTANT,
    BasicHumanValues.TRADITION: Likert5ImportanceLevel.OF_SUPREME_IMPORTANCE,
}

cognitive_biases = [CognitiveBias.ADDITIVE_BIAS, CognitiveBias.AGENT_DETECTION]

fallacies = [Fallacy.AD_HOMINEM, Fallacy.APPEAL_TO_AUTHORITY]

vote_last_presidential_election = "voted for Donald Trump"

ideologies = {
    Issues.ECONOMIC: Ideology.MODERATE,
    Issues.SOCIAL: Ideology.CONSERVATIVE,
}

agreement_with_statements = {
    "abortions should be illegal": Likert7AgreementLevel.NEUTRAL,
    "guns should be banned": Likert7AgreementLevel.NEUTRAL,
    "the death penalty should be abolished": Likert7AgreementLevel.SLIGHTLY_AGREE,
}

likelihood_of_beliefs = {
    "the Earth is flat": Likert11LikelihoodLevel.NEUTRAL,
    "vaccines cause autism": Likert11LikelihoodLevel.NEUTRAL,
    "climate change is a hoax": Likert11LikelihoodLevel.SOMEWHAT_UNLIKELY,
}

free_form_opinions = ["immigration is a problem", "the government is corrupt"]

personality = Personality(
    demographic_profile=demo_profile,
    traits=traits,
    facets=facets,
    moral_foundations=moral_foundations,
    basic_human_values=basic_human_values,
    cognitive_biases=cognitive_biases,
    fallacies=fallacies,
    vote_last_presidential_election=vote_last_presidential_election,
    ideologies=ideologies,
    agreement_with_statements=agreement_with_statements,
    likelihood_of_beliefs=likelihood_of_beliefs,
    free_form_opinions=free_form_opinions,
)

DebaterConfig(
    name="Bob",
    position=TopicOpinion(
        agreement=Likert7AgreementLevel.STRONGLY_DISAGREE),
    personality=personality
    variable_topic_opinion=False,
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

Note: you can ignore the summary and only include the latest `N` messages in the debater prompts by setting `SummaryConfig.ignore` to `True` (`False` by default).

The mediator prompts will only include the latest `N` messages and not the summaries, as in real-life situations, a responsive LLM mediator setup will not have the time to recompute a real-time summary of ongoing chat exchanges before intervening.

The summary config also includes a list of the debaters in order to include the names of every debater before the summary in LLM prompts.

```python
from llm_mediator_simulation.simulation.summary.config import SummaryConfig

summary_config = SummaryConfig(latest_messages_limit=3, debaters=debaters)
```

You do not need to manually create a `SummaryHandler`.

### Debate

Defined in the [`simulation/debate`](./src/llm_mediator_simulation/simulation/debate) directory.

The debate configuration only includes the debated statement, a short situation context prompt, and prompts to include for debaters according to their position regarding the current debate's statement.
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

5 model wrappers are defined:

- Gemini models (suitable for mediator simulations).
- GPT models (suitable for mediator simulations).
- Mistral models (remote and local, suitable for both).
- Models served by a local Ollama server (suitable for both).

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

## Running the debate

You can now run a debate simulation!

You just need to create a `DebateHandler` instance and pass it all the configuration needed:

```python
from llm_mediator_simulation.simulation.debate.handler import DebateHandler

debate = DebateHandler(
    debater_model=debater_model,
    mediator_model=mediator_model,
    debaters=debaters,
    config=debate_config,
    summary_config=summary_config,
    metrics_handler=metrics,
    mediator_config=mediator_config, # Set it to None if you want to experiment without a mediator
)
```

You can then run multiple rounds of the debate, and save the results in a pickle file for later analysis:

```python
debate.run(rounds=3)

# Saved to ./debate_archive.pkl
debate.pickle("debate_archive")
```

Note that even after saving the debate to a pickle archive, you can continue running rounds.

## Analysis

A script is provided to analyze pickled debate files at [examples/example_analysis.py](./examples/example_analysis.py)

The following commands are available:

(Note: if you replace the `debate.pkl` argument with a directory `debate_dir/`, the script will use the last debate in the directory.)


Plot the metrics of a debate:

```bash
python examples/example_analysis.py metrics -d debate.pkl
python examples/example_analysis.py metrics -d debate.pkl -a  # Averaged over debaters
```

Plot the personalities of debaters over time:

```bash
python examples/example_analysis.py personalities -d debate.pkl
python examples/example_analysis.py personalities -d debate.pkl -a  # Averaged over debaters
```

Generate a transcript of the debate:

```bash
python examples/example_analysis.py transcript -d debate.pkl
```

Generate the transcript of the last debate in the debates_dir directory
```bash
python examples/example_analysis.py transcript -d debates_dir/ 
```

Print the debate data in a pretty format:

```bash
python examples/example_analysis.py print -d debate.pkl
```

You can save the pretty print on disk and preview it if you install the [ANSI Colors VSCode](https://marketplace.visualstudio.com/items?itemName=iliazeus.vscode-ansi) extension (with ⌘⇧V on Mac or Ctrl+K V on Windows/Linux).

```bash
python examples/example_analysis.py print -d debate.pkl > debates_sandbox/output.ans
```
![screenshot to preview the pretty print](https://github.com/iliazeus/vscode-ansi/raw/HEAD/images/screenshot-editorTitleButton-darkPlus.png)

## Other features

### Benchmarks

LLM API calls can be benchmarked for speed.
Functions annotated with the `@benchmark` decorator have their call duration stored in the global `BENCHMARKS` object.
The implementation is in the [`utils/decorators`](./src/llm_mediator_simulation/utils/decorators.py) file.

### Integration with deliberate-lab.appspot.com

CSV debate transcripts from the `deliberate-lab.appspot.com` app can be imported into a debate handler for further simulation.
Note that debater personalities and positions cannot be inferred from this transcript yet, so the simulation may not reflect accurately the original debater personalities. If you want to continue a proper simulation, you may want to import the chat data first, and then assign personalities to the debater configs stored in the `initial_debaters` field of the `DebateHandler` class.

See the `preload_csv_chat` method from the [`DebateHandler`](./src/llm_mediator_simulation/simulation/debate/handler.py) class.

### Example scripts

Multiple example scripts are provided in the [`examples`](./examples) directory.

- [`example_analysis.py`](./examples/example_analysis.py): cli helper to analyze debate data.
- [`example_async.py`](./examples/example_async.py): run multiple debate simulations asynchronously.
- [`example_debate.py`](./examples/example_debate.py): run a single debate simulation.
- [`example_mistral.py`](./examples/example_mistral.py): test script to run the mistral 7B model locally.
- [`example_server.py`](./examples/example_server.py): run a local mistral model inside a web server to experiment with LLM calls without having to wait for model initialization inbetween calls.

### Getting started
Save GPT's API key in a `.env` file at the root of the project.

```bash
python examples/example_server.py start # Start the local mistral model server
python scripts/default_debate_server.py # One Local Mistral LLM queried as a server, responding to debater prompts; One GPT's ``mediator'' acting only as a summarizer (no mediator intervention) and plot the transcript
python examples/example_analysis.py transcript -d debates_sandbox/my_debate.pkl # Plot the transcript of the my_debate
```

### Streamlit app
Run the Streamlit app with the following command:

```bash
streamlit run app/home.py
```
See [here](here) for a demo of short demo of the app.