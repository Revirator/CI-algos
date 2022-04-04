from MyEGreedy import MyEGreedy
from MyQLearning import MyQLearning

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from Agent import Agent
from Maze import Maze

import random

if __name__ == "__main__":
    # TOY MAZE

    file = "../../data/toy_maze.txt"
    maze = Maze(file)

    goal_1 = maze.get_state(9, 9)
    maze.set_reward(goal_1, 10)
    goals = [goal_1]

    # SECOND GOAL FOR TOY MAZE

    # goal_2 = maze.get_state(9, 0)
    # goals = [goal_1, goal_2]
    # maze.set_reward(goal_2, 5)

    # --------------------------------

    # EASY MAZE

    # file = "../../data/easy_maze.txt"
    # maze = Maze(file)
    #
    # goal = maze.get_state(24, 14)
    # maze.set_reward(goal, 10)
    # goals = [goal]

    # --------------------------------

    # Create a robot at starting and reset location (0,0) (top left)
    robot = Agent(0, 0)

    # Make a selection object (you need to implement the methods in this class)
    selection = MyEGreedy()

    # Make a Qlearning object (you need to implement the methods in this class)
    learn = MyQLearning()

    # HYPER-PARAMETERS
    epsilon = 0.1
    alpha = 0.7
    gamma = 0.9

    # random.seed(42)

    # Keep learning until the stopping criterion is met
    steps_taken = 0
    terminate = False
    while not terminate:
        # Proceed stepping until we reach the terminating state or 30000 steps
        while True:
            state = robot.get_state(maze)
            # Get the next action for this step
            action = selection.get_egreedy_action(robot, maze, learn, epsilon)
            # Do the action
            new_state = robot.do_action(action, maze)
            # Update Q
            learn.update_q(state, action, maze.get_reward(new_state), new_state, maze.get_valid_actions(robot), alpha, gamma)
            # Update the number of steps taken
            steps_taken += 1
            if robot.get_state(maze) in goals:
                break
            if steps_taken >= 30000:
                terminate = True
                break
        robot.reset()

    while True:
        state = robot.get_state(maze)
        action = selection.get_best_action(robot, maze, learn)
        robot.do_action(action, maze)
        if robot.get_state(maze) in goals:
            break
    print(f"Solution: {robot.nr_of_actions_since_reset}")
    robot.reset()
