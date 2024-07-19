"""Single-run test for local mistral model."""

from llm_mediator_simulation.models.mistral_local_model import MistralLocalModel

model = MistralLocalModel(max_length=500)

prompt = "Hello, how are you?"

print(model.sample(prompt))
