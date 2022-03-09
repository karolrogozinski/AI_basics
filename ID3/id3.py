from typing import List

import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split


class Node:
    """
    Parent class for Leaf and Root.
    Contains method that returns value.
    """
    def __init__(self, value: str = None):
        self._value = value

    def get_value(self) -> str:
        return self._value

    def is_id3(self) -> bool:
        return False


class Branch:
    """
    Connection between noodes

    Attributes:
        value (str) : value directing to next node
        node (Leaf/ID3) : next node
    """
    def __init__(self, value : str, node: Node) -> None:
        self._value = value
        self._node = node

    def get_value(self) -> str:
        return self._value

    def get_node(self) -> Node:
        return self._node


class Leaf(Node):
    """
    Class representing leaf of decision tree
    """
    def __init__ (self, value: str):
        super().__init__(value)

    def is_leaf(self) -> bool:
        return True


class Root(Node):
    """
    Class representing root of ID3
    """
    def __init__(self, value: str):
        super().__init__(value)
        self._branches = []

    def add_branch(self, branch: Branch) -> None:
        self._branches.append(branch)

    def get_branches(self) -> List[Branch]:
        return list(self._branches)

    def is_leaf(self) -> bool:
        return False


class ID3:
    """
    Class representing decision tree built by ID3 algorithm
    """
    def __init__(self) -> None:
        self.root = None
        self.most_frequent_class = None

    def is_id3(self) -> bool:
        return True

    def get_value(self) -> str:
        """
        Returns value of the root
        """
        return self.root.get_value()

    def get_branches(self) -> List:
        """
        Returns branches of the root
        """
        return self.root.get_branches()

    def fit(self, data: pd.DataFrame, class_name: str) -> None:
        """
        Train ID3 on given data

        Argumentss:
            data (DataFrame): data
            class_name (str): name of column to predict
        """
        # End with leaf if conditions satisfied
        if data[class_name].value_counts().size == 1:
            self.root = Leaf(data[class_name].value_counts().idxmax())
            return
        
        if pd.DataFrame(data).columns.size == 1:
            self.root = Leaf(data[class_name].value_counts().idxmax())
            return

        best_attr = self.find_best_attribute(data, class_name)
        self.root = Root(best_attr)
        self.most_frequent_class = data[class_name].value_counts().idxmax()

        # Create subtrees
        for value in data[best_attr].unique():
            sub_id3 = ID3()
            self.root.add_branch(Branch(value, sub_id3))
            sub_id3.fit(
                data.loc[data[best_attr] == value].drop(columns=best_attr),
                class_name
                )

    def predict(self, data: pd.DataFrame) -> List[str]:
        """
        Predict classes of given dataset

        Arguments:
            data (DataFrame): data

        Returns:
            List[str] : list of predicted classes
        """
        results = []
        for index, row in data.iterrows():
            curr_node = self.root
            flag = True
            while True:
                # if current value doesnt match any branch
                if not flag:
                    results.append(self.most_frequent_class)
                    flag = True
                    break
                flag = False
                if curr_node.is_id3():
                    curr_node = curr_node.root
                if curr_node.is_leaf():
                    results.append(curr_node.get_value())
                    flag = True
                    break
                for branch in curr_node.get_branches():
                    if row[curr_node.get_value()] == branch.get_value():
                        curr_node = branch.get_node()
                        flag = True
                        break
        return results

    def classify_csv(self, filename: str, class_name: str) -> pd.DataFrame:
        """
        Classify data from csv file

        Args:
            filename (str): name of csv file
            class_name (str): name of class column

        Returns:
            pd.DataFrame: Actual and predicted values
        """
        data = pd.read_csv(filename)
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=[class_name]),
            data[class_name],
            test_size = 0.4,
            random_state = 42
        )

        train_data = X_train
        train_data[class_name] = y_train
        self.fit(train_data, class_name)

        y_data = pd.DataFrame()
        y_data['y'] = y_test
        y_data['y_pred'] = self.predict(X_test)

        print('Accuracy: ', self.accuracy(y_data))
        return y_data
    

    @staticmethod
    def accuracy(data: pd.DataFrame):
        """
        Calculates accuracy between dataframe with 2 columns
        """
        return sum(
            np.where(data.iloc[:, 0] == data.iloc[:, 1], True, False)
            ) / data.shape[0]

    @staticmethod
    def find_best_attribute(data: pd.DataFrame, class_name: str) -> str:
        """
        Calculate information gains for all attributes and find best one.

        Arguments:
            data (DataFrame) : df to evaluate
            class_name (str) : name of the class column

        Returns:
            (str) : name of the best attribute
        """
        col_names = data.drop(columns=[class_name]).columns.tolist()

        return max(
            col_names,
            key=lambda name: inf_gain(data[[name, class_name]])
        )


def inf_gain(learning_pairs: pd.DataFrame) -> float:
    """
    Calculate information gain of given argument column.

    Arguments:
        learning_pairs (DataFrame) : df with 2 columns: arguments and classes

    Returns:
        (float) : information gain of given column
    """
    column_names = learning_pairs.columns.tolist()
    x_name, y_name = column_names[0], column_names[1]

    U_entropy = entropy(learning_pairs[y_name].value_counts())

    # calculate weight and entropy for each unique value in arguments 
    args_data = learning_pairs.groupby(x_name)[y_name].apply(
        lambda x :
        (x.value_counts().sum() / learning_pairs[x_name].value_counts().sum(),
        entropy(x.value_counts()))
    )

    return U_entropy - sum(arg[0] * arg[1] for arg in args_data)
