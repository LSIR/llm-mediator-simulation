import unittest
import csv
import os
from tempfile import NamedTemporaryFile

from llm_mediator_simulation.personalities.cognitive_biases import CognitiveBias


class TestCognitiveBiasLoading(unittest.TestCase):

    def setUp(self):
        # if data/cognitive_biases.csv exists
        default_path = 'data/cognitive_biases.csv'
        if os.path.exists(default_path):
            self.csv_path = default_path
        else:
            # Create a temporary CSV file
            self.temp_csv = NamedTemporaryFile(delete=False, mode='w', newline='', encoding='utf-8')
            self.csv_path = self.temp_csv.name
            # Write headers and sample data into the CSV file
            fieldnames = ['Group', 'Name', 'Type', 'Description']
            writer = csv.DictWriter(self.temp_csv, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({
                'Group': 'Belief, decision-making and behavioral',
                'Name': 'Additive bias',
                'Type': '',
                'Description': 'The tendency to solve problems through addition, even when subtraction is a better approach.'
            })
            writer.writerow({
                'Group': 'Belief, decision-making and behavioral',
                'Name': 'Agent detection',
                'Type': 'False priors',
                'Description': 'The inclination to presume the purposeful intervention of a sentient or intelligent agent.'
            })
            self.temp_csv.close()


    def test_load_cognitive_biases(self):

        # Test if the CognitiveBias Enum was correctly populated
        self.assertTrue(hasattr(CognitiveBias, 'ADDITIVE_BIAS'))
        self.assertTrue(hasattr(CognitiveBias, 'AGENT_DETECTION'))

        # Check sample data integrity
        additive_bias = CognitiveBias.ADDITIVE_BIAS
        self.assertEqual(additive_bias.name, 'Additive bias')
        self.assertEqual(additive_bias.group, 'Belief, decision-making and behavioral')
        self.assertEqual(additive_bias.type, '')
        self.assertEqual(additive_bias.description, 'The tendency to solve problems through addition, even when subtraction is a better approach.')

        agent_detection = CognitiveBias.AGENT_DETECTION
        self.assertEqual(agent_detection.name, 'Agent detection')
        self.assertEqual(agent_detection.group, 'Belief, decision-making and behavioral')
        self.assertEqual(agent_detection.type, 'False priors')
        self.assertEqual(agent_detection.description, 'The inclination to presume the purposeful intervention of a sentient or intelligent agent.')

if __name__ == '__main__':
    unittest.main()