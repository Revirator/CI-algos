import numpy as np

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from QLearning import QLearning


class MyQLearning(QLearning):

    def update_q(self, state, action, r, state_next, possible_actions, alpha, gamma):
        action_values = self.get_action_values(state_next, possible_actions)
        q_max = action_values[np.argmax(action_values)]
        q_old = self.get_q(state, action)
        q_new = q_old + alpha * (r + gamma * q_max - q_old)
        self.set_q(state, action, q_new)
