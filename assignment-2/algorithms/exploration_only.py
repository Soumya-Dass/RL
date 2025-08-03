import numpy as np
import random

class ExplorationOnly:
    def __init__(self, n_arms, *args, **kwargs):
        self.n_arms = n_arms
        self.estimates = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms)

    def select_arm(self):
        return random.randrange(self.n_arms)

    def update(self, chosen_arm, reward):
        self.pulls[chosen_arm] += 1
        # Note: We don't update estimates for this algorithm

    # ADDED: Method to satisfy the testing framework
    def get_estimated_optimal_arm(self):
        """Returns the arm with the highest current estimate."""
        return int(np.argmax(self.estimates))

    def reset(self):
        self.estimates = np.zeros(self.n_arms)
        self.pulls = np.zeros(self.n_arms)