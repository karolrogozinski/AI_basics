from random import uniform

import numpy as np
import gym


def default_reward(state: str) -> int:
    """
    Default reward system - just 1 for goal and 0 in other cases.
    """
    if state == b'G':
        return 1
    return 0


def dont_like_holes_reward(state: str) -> int:
    """
    Default reward system extended by punishment for holes
    """
    if state == b'G':
        return 1
    elif state == b'H':
        return -1
    return 0


def dont_go_back_reward(state: str) -> int:
    """
    Dont back to start
    """
    if state == b'G':
        return 1
    elif state == b'H':
        return -1
    elif state == b'S':
        return -1
    return 0


class Qlearning:
    """
    Just Q-Learning algorithm
    """
    def __init__(self, env_name: str = "FrozenLake8x8-v1") -> None:
        self.env = gym.make(env_name)
        self.q_matrix = np.zeros(shape = (
            self.env.nrow, self.env.ncol, self.env.action_space.n)
            )
        self.possible_moves = np.array(range(self.env.action_space.n))

    def get_q(self, state: int, move: int) -> float:
        row, col = self.to_index(state)
        return self.q_matrix[int(row)][col][move]

    def to_index(self, state: int) -> tuple():
        col = state % self.env.ncol
        row = (state - col) / self.env.ncol
        return int(row), int(col)

    def to_state(self, row: int, col: int) -> int:
        return row * self.env.ncol + col

    def find_best(self, state: int) -> float:
        if self.env.desc.flatten()[self.state] in b'HG':
            return 0
        row, col = self.to_index(state)
        return np.amax(self.q_matrix[int(row)][col])

    def check_end(self, state: int) -> int:
        field = self.env.desc.flatten()[state]
        if field == b'H':
            return 1
        elif field == b'G':
            return 2
        return 0

    def step_state(self, move: int) -> int:
        row, col = self.to_index(self.state)
        if move == 0:
            col = max(col - 1, 0)
        elif move == 1:
            row = min(row + 1, self.env.nrow - 1)
        elif move == 2:
            col = min(col + 1, self.env.ncol - 1)
        elif move == 3:
            row = max(row - 1, 0)
        state_ = self.to_state(row, col)
        return state_

    def train_and_evaluate(
            self,
            iterations: int = 1000,
            max_moves: int = 200,
            leaning_rate: float = 0.1,
            discount_rate: float = 0.1,
            epsilon: float = 0.01,
            reward_function: callable = default_reward
    ):
        goals = 0
        for i in range(iterations):
            self.state = self.env.reset()
            for _ in range(max_moves):
                # find best move with epsilon greedy
                np.random.shuffle(self.possible_moves)
                if uniform(0, 1) < epsilon:
                    next_move = self.possible_moves[0]
                else:
                    next_move = max(
                        self.possible_moves,
                        key=lambda move: (
                                self.get_q(self.step_state(move), move)
                        )
                    )
                # move 
                next_state, _, _, _ = self.env.step(next_move)
                reward = reward_function(self.env.desc.flatten()[next_state])
                # update q
                row, col = self.to_index(self.state)
                max_q = self.find_best(next_state)
                self.q_matrix[row][col][next_move] += leaning_rate * (
                    reward + discount_rate * max_q - self.q_matrix[row][col][next_move]
                )
                if self.check_end(next_state):
                    goals += self.check_end(next_state) - 1
                    break

                self.state = next_state
        # return accuracy
        return goals / iterations


reward_systems = [default_reward, dont_like_holes_reward, dont_go_back_reward]
for reward in reward_systems:
    results = list()
    for _ in range(500):
        agent = Qlearning()
        result = agent.train_and_evaluate(reward_function=reward)
        results.append(result)

    results = np.array(results)
    to_print = dict()
    to_print["Min"] = np.amin(results)
    to_print["Max"] = np.amax(results)
    to_print["Std"] = np.std(results)
    to_print["Mean"] = np.mean(results)
    to_print["Median"] = np.median(results)

    print("\n", reward.__name__, "\n", to_print, "\n")
