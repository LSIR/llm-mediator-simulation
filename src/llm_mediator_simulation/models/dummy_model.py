from typing import override
from llm_mediator_simulation.models.language_model import LanguageModel

class DummyModel(LanguageModel):  
    def __init__(self, model_name: str ='dummy_model') -> None:
        self.model_name = model_name


    @override
    def sample(self, prompt: str, seed: int | None = None) -> str:

        return f"""```json
                    {{
                        "do_intervene": true,
                        "intervention_justification": "dummy justification",
                        "text": "{str(hash(prompt))}"
                    }}
                    ```
                """