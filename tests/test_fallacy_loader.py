import unittest
import csv
import os
from tempfile import NamedTemporaryFile

from llm_mediator_simulation.personalities.fallacies import Fallacy


class TestFallacyLoading(unittest.TestCase):
    def setUp(self):
        # if data/fallacies.json exists
        default_path = "data/fallacies.json"
        if os.path.exists(default_path):
            self.csv_path = default_path
        else:
            # Create a temporary JSON file
            self.temp_json = NamedTemporaryFile(
                delete=False, mode="w", newline="", encoding="utf-8"
            )
            self.json_path = self.temp_json.name
            # Write sample data into the JSON file
            self.temp_json.write("""[{"name": "Ad Hominem", 
                                      "definition": "Attacking the arguer instead of the argument."}, 
                                     {"name": "Affirmative Conclusion from a Negative Premise", 
                                      "definition": "A categorical syllogism has a positive conclusion, but at least one negative premise."}]""")
            self.temp_json.close()

    def test_load_fallacies(self):
        # Test if the CognitiveBias Enum was correctly populated
        self.assertTrue(hasattr(Fallacy, "AD_HOMINEM"))
        self.assertTrue(
            hasattr(Fallacy, "AFFIRMATIVE_CONCLUSION_FROM_A_NEGATIVE_PREMISE")
        )

        # Check sample data integrity
        add_hominem = Fallacy.AD_HOMINEM
        self.assertEqual(add_hominem.name, "Ad Hominem")
        self.assertEqual(
            add_hominem.description, "Attacking the arguer instead of the argument."
        )

        affirmative_conclusion_from_a_negative_premise = (
            Fallacy.AFFIRMATIVE_CONCLUSION_FROM_A_NEGATIVE_PREMISE
        )
        self.assertEqual(
            affirmative_conclusion_from_a_negative_premise.name,
            "Affirmative Conclusion from a Negative Premise",
        )
        self.assertEqual(
            affirmative_conclusion_from_a_negative_premise.description,
            "A categorical syllogism has a positive conclusion, but at least one negative premise.",
        )


if __name__ == "__main__":
    unittest.main()
