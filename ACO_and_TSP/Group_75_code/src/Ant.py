import os
import random
import sys

import numpy as np

from Route import Route
from Direction import Direction

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


# Class that represents the ants' functionality.
class Ant:

    # Constructor for ant taking a Maze and PathSpecification.
    # @param maze Maze the ant will be running in.
    # @param spec The path specification consisting of a start coordinate and an end coordinate.
    def __init__(self, maze, path_specification):
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
        self.visited = np.zeros(self.maze.pheromone_matrix.shape)
        self.rand = random

    # Method that performs a single run through the maze by the ant.
    # @return The route the ant found through the maze.
    def find_route(self):
        route = Route(self.start)
        self.visited[self.start.get_y(), self.start.get_x()] = 1
        while True:
            if self.current_position == self.end:
                break
            surrounding_pheromone = self.maze.get_surrounding_pheromone(self.current_position)
            total_pheromone = surrounding_pheromone.get_total_surrounding_pheromone()

            # Prevent ant from going back to coordinate it has already visited
            east_pheromone = (surrounding_pheromone.get(Direction.east) / total_pheromone)
            if east_pheromone != 0:
                if self.visited[self.current_position.get_y(), self.current_position.get_x() + 1] == 1:
                    east_pheromone *= 0
            north_pheromone = (surrounding_pheromone.get(Direction.north) / total_pheromone)
            if north_pheromone != 0:
                if self.visited[self.current_position.get_y() - 1, self.current_position.get_x()] == 1:
                    north_pheromone *= 0
            west_pheromone = (surrounding_pheromone.get(Direction.west) / total_pheromone)
            if west_pheromone != 0:
                if self.visited[self.current_position.get_y(), self.current_position.get_x() - 1] == 1:
                    west_pheromone *= 0
            south_pheromone = (surrounding_pheromone.get(Direction.south) / total_pheromone)
            if south_pheromone != 0:
                if self.visited[self.current_position.get_y() + 1, self.current_position.get_x()] == 1:
                    south_pheromone *= 0

            if sum([east_pheromone, north_pheromone, west_pheromone, south_pheromone]) == 0:
                # Reached a dead end. Go back!
                self.current_position = self.current_position.subtract_direction(route.remove_last())
            else:
                prob = np.array([east_pheromone, north_pheromone, west_pheromone, south_pheromone])
                dirs = np.array([Direction.east, Direction.north, Direction.west, Direction.south])

                # Take direction of max pheromone 12.5% of the time
                if self.rand.uniform(0, 1) >= 0.875:
                    new_dir = dirs[np.argmax(prob)]
                else:
                    new_dir = self.rand.choices(list(dirs), list(prob), k=1)[0]
                route.add(new_dir)
                self.current_position = self.current_position.add_direction(new_dir)
            self.visited[self.current_position.get_y(), self.current_position.get_x()] = 1
        return route
