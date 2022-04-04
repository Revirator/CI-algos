import random
import numpy as np


class MyEGreedy:

    def __init__(self):
        print("Made EGreedy")

    def get_random_action(self, agent, maze):
        valid_actions = maze.get_valid_actions(agent)
        return random.choice(valid_actions)

    def get_best_action(self, agent, maze, q_learning):
        valid_actions = maze.get_valid_actions(agent)
        action_values = q_learning.get_action_values(agent.get_state(maze), valid_actions)

        # If all action values are equal - choose the action randomly
        # This ensures that the agent is not biased towards selecting
        # the same action over and over.
        if action_values.count(action_values[0]) == len(action_values):
            return self.get_random_action(agent, maze)
        else:
            return valid_actions[np.argmax(action_values)]

    def get_egreedy_action(self, agent, maze, q_learning, epsilon):
        if random.uniform(0, 1) <= epsilon:
            return self.get_random_action(agent, maze)
        else:
            return self.get_best_action(agent, maze, q_learning)
