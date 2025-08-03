import numpy as np

class UCB:
    def __init__(self, n_arms, c=2, **kwargs):
        self.n_arms = n_arms
        self.c = c
        self.estimates = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms)
        self.total_pulls = 0

    def select_arm(self):
        unpulled_arms = np.where(self.pulls == 0)[0]
        if len(unpulled_arms) > 0:
            return int(unpulled_arms[0])

        bonus = self.c * np.sqrt(np.log(self.total_pulls) / (self.pulls + 1e-5))
        ucb_values = self.estimates + bonus
        
        return int(np.argmax(ucb_values))

    def update(self, chosen_arm, reward):
        self.pulls[chosen_arm] += 1
        self.total_pulls += 1
        
        pull_count = self.pulls[chosen_arm]
        old_estimate = self.estimates[chosen_arm]
        self.estimates[chosen_arm] = old_estimate + (1 / pull_count) * (reward - old_estimate)

    # ADDED: Method to satisfy the testing framework
    def get_estimated_optimal_arm(self):
        """Returns the arm with the highest current estimate (not UCB value)."""
        return int(np.argmax(self.estimates))

    def reset(self):
        self.estimates = np.zeros(self.n_arms)
        self.pulls = np.zeros(self.n_arms)
        self.total_pulls = 0