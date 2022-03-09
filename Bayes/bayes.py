import json
import random

import pandas as pd

from id3 import ID3


class BayesGenerator:
    """
    Generator of random samples, based on Bayes Net
    """
    def __init__(self, file_name: str) -> None:
        self.nodes = []
        self.names = []
        with open(file_name, "r") as file:
            self.create_structure(json.load(file))

    class Node:
        """
        Single node in bayes structure
        """
        def __init__(self, name: str, causes: list, probs: dict):
            self.name = name
            self.causes = causes
            self.probs = probs

    def get_node(self, name: str) -> Node:
        """
        Return node with given name
        """
        for node in self.nodes:
            if node.name == name:
                return node

    def create_structure(self, json_data: set) -> None:
        """
        Create node structure and set column names
        """
        for sample in json_data:
            name = sample["name"]
            self.names.append(name)
            causes = []
            if sample["causes"]:
                for cause in sample["causes"]:
                    causes.append(self.get_node(cause))
            probs = sample["prob_table"]
            self.nodes.append(self.Node(name, causes, probs))

    def get_column(self, col_name: str) -> int:
        """
        Returns number of column with given name
        """
        for idx, name in enumerate(self.names):
            if name == col_name:
                return idx

    def calc_prob(self, name_: str, sample: list) -> float:
        """
        Calculate probability for given argument
        """
        node = self.get_node(name_)
        if not node.causes:
            return node.probs[""]
        cause_str = ""
        for cause in node.causes:
            cause_str += sample[self.get_column(cause.name)]
        return node.probs[cause_str]

    def generate_sample(self) -> list:
        """
        One row in our generated data
        """
        sample = []
        for node in self.nodes:
            if random.uniform(0, 1) < self.calc_prob(node.name, sample):
                sample.append("1")
            else:
                sample.append("0")
        return sample

    def generate_df(self, samples_num: int) -> None:
        """
        Generate given number of samples, based on structure
        """
        data = []
        for _ in range(samples_num):
            data.append(self.generate_sample())
        df = pd.DataFrame(data, columns=self.names)
        df.to_csv('samples.csv', index=False)


bg = BayesGenerator("data.json")
bg.generate_df(10000)

model = ID3()
model.classify_csv('samples.csv', 'Ache')
