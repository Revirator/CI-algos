import os
import sys
import traceback
import numpy as np

from SurroundingPheromone import SurroundingPheromone
from Direction import Direction

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


# Class that holds all the maze data. This means the pheromones, the open and blocked tiles in the system as
# well as the starting and end coordinates.
class Maze:
    # Constructor of a maze
    # @param walls int array of tiles accessible (1) and non-accessible (0)
    # @param width width of Maze (horizontal)
    # @param length length of Maze (vertical)
    def __init__(self, walls, width, length):
        self.walls = walls
        self.length = length
        self.width = width
        self.start = None
        self.end = None
        self.pheromone_matrix = None
        self.initialize_pheromones()

    # Initialize pheromones to a start value.
    def initialize_pheromones(self):
        self.pheromone_matrix = np.array(self.walls, dtype=float).T
        for i in range(1, self.length - 1):
            for j in range(1, self.width - 1):
                if self.pheromone_matrix[i][j] == 0:
                    continue
                min_length = sys.maxsize
                for k in range(j - 1, -1, -1):
                    if self.pheromone_matrix[i][k] == 0:
                        if j - k < min_length:
                            min_length = j - k
                        break
                    elif self.pheromone_matrix[i - 1][k] == 0 or self.pheromone_matrix[i + 1][k] == 0:
                        if j - k + 1 < min_length:
                            min_length = j - k + 1
                        break
                if j < min_length:
                    min_length = j
                for k in range(j + 1, self.width):
                    if self.pheromone_matrix[i][k] == 0:
                        if k - j < min_length:
                            min_length = k - j
                        break
                    elif self.pheromone_matrix[i - 1][k] == 0 or self.pheromone_matrix[i + 1][k] == 0:
                        if k - j + 1 < min_length:
                            min_length = k - j + 1
                        break
                for k in range(i - 1, -1, -1):
                    if self.pheromone_matrix[k][j] == 0:
                        if i - k < min_length:
                            min_length = i - k
                        break
                    elif self.pheromone_matrix[k][j - 1] == 0 or self.pheromone_matrix[k][j + 1] == 0:
                        if i - k + 1 < min_length:
                            min_length = i - k + 1
                        break
                if i < min_length:
                    min_length = i
                for k in range(i + 1, self.length):
                    if self.pheromone_matrix[k][j] == 0:
                        if k - i < min_length:
                            min_length = k - i
                        break
                    elif self.pheromone_matrix[k][j - 1] == 0 or self.pheromone_matrix[k][j + 1] == 0:
                        if k - i + 1 < min_length:
                            min_length = k - i + 1
                        break
                if min_length == sys.maxsize:
                    min_length = 1
                self.pheromone_matrix[i][j] /= min_length

    # Reset the maze for a new shortest path problem.
    def reset(self):
        self.initialize_pheromones()

    # Update the pheromones along a certain route according to a certain Q
    # @param r The route of the ants
    # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_route(self, route, q):
        value = q / max(1, route.size())
        pos = route.get_start()
        for direction in route.get_route():
            pos = pos.add_direction(direction)
            self.pheromone_matrix[pos.get_y(), pos.get_x()] += value

    # Update pheromones for a list of routes
    # @param routes A list of routes
    # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_routes(self, routes, q):
        for r in routes:
            self.add_pheromone_route(r, q)

    # Evaporate pheromone
    # @param rho evaporation factor
    def evaporate(self, rho):
        assert 0 <= rho <= 1
        self.pheromone_matrix *= (1 - rho)

    # Width getter
    # @return width of the maze
    def get_width(self):
        return self.width

    # Length getter
    # @return length of the maze
    def get_length(self):
        return self.length

    # Pheromone getter for a specific position. If the position is not in bounds returns 0
    # @param pos Position coordinate
    # @return pheromone at point
    def get_pheromone(self, pos):
        if self.in_bounds(pos):
            return self.pheromone_matrix[pos.get_y(), pos.get_x()]
        return 0

    # Returns the amount of pheromones on the neighbouring positions (N/E/S/W).
    # @param position The position to check the neighbours of.
    # @return the pheromones of the neighbouring positions.
    def get_surrounding_pheromone(self, pos):
        if self.in_bounds(pos):
            return SurroundingPheromone(self.get_pheromone(pos.add_direction(Direction.north)),
                                        self.get_pheromone(pos.add_direction(Direction.east)),
                                        self.get_pheromone(pos.add_direction(Direction.south)),
                                        self.get_pheromone(pos.add_direction(Direction.west)))
        return None

    # Check whether a coordinate lies in the current maze.
    # @param position The position to be checked
    # @return Whether the position is in the current maze
    def in_bounds(self, pos):
        return pos.x_between(0, self.width) and pos.y_between(0, self.length)

    # Representation of Maze as defined by the input file format.
    # @return String representation
    def __str__(self):
        string = ""
        string += str(self.width)
        string += " "
        string += str(self.length)
        string += " \n"
        for y in range(self.length):
            for x in range(self.width):
                string += str(self.walls[x][y])
                string += " "
            string += "\n"
        return string

    # Method that builds a maze from a file
    # @param filePath Path to the file
    # @return A maze object with pheromones initialized to 0's inaccessible and 1's accessible.
    @staticmethod
    def create_maze(file_path):
        try:
            f = open(file_path, "r")
            lines = f.read().splitlines()
            dimensions = lines[0].split(" ")
            width = int(dimensions[0])
            length = int(dimensions[1])

            # make the maze_layout
            maze_layout = []
            for x in range(width):
                maze_layout.append([])

            for y in range(length):
                line = lines[y + 1].split(" ")
                for x in range(width):
                    if line[x] != "":
                        state = int(line[x])
                        maze_layout[x].append(state)
            print("Ready reading maze file " + file_path)
            return Maze(maze_layout, width, length)
        except FileNotFoundError:
            print("Error reading maze file " + file_path)
            traceback.print_exc()
            sys.exit()
