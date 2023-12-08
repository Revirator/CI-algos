import os
import sys
import time

from matplotlib import pyplot as plt

from Ant import Ant
from Maze import Maze
from PathSpecification import PathSpecification

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


# Class representing the first assignment. Finds the shortest path between two points in a maze according to a specific
# path specification.
class AntColonyOptimization:
    # Constructs a new optimization object using ants.
    # @param maze the maze.
    # @param antsPerGen the amount of ants per generation.
    # @param generations the amount of generations.
    # @param Q normalization factor for the amount of dropped pheromone.
    # @param evaporation the evaporation factor.
    def __init__(self, maze, ants_per_gen, generations, q, evaporation):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation

    # Loop that starts the shortest path process
    # @param spec Specification of the route we wish to optimize
    # @return ACO optimized route
    def find_shortest_route(self, path_specification):
        self.maze.reset()
        min_route = None
        for _ in range(self.generations):
            routes = []
            for _ in range(self.ants_per_gen):
                ant = Ant(self.maze, path_specification)
                r = ant.find_route()
                if min_route is None:
                    min_route = r
                elif r.shorter_than(min_route):
                    min_route = r
                routes.append(r)
            self.maze.evaporate(self.evaporation)
            # Sort on size and only take the shortest 33%
            routes.sort(key=lambda ro: ro.size())
            routes = routes[:int(len(routes) // 3)]
            self.maze.add_pheromone_routes(routes, self.q)
        # plt.imshow(self.maze.pheromone_matrix, interpolation="nearest")
        # plt.show()
        return min_route

    @staticmethod
    def aco_easy(gen, no_gen, q, evap):
        maze = Maze.create_maze("./../data/easy maze.txt")
        spec = PathSpecification.read_coordinates("./../data/easy coordinates.txt")
        aco = AntColonyOptimization(maze, gen, no_gen, q, evap)

        start_time = int(round(time.time() * 1000))
        shortest_route = aco.find_shortest_route(spec)
        print("Time taken (easy): " + str((int(round(time.time() * 1000)) - start_time) / 1000.0))
        shortest_route.write_to_file("./../results/easy_solution.txt")
        print("Route size (easy): " + str(shortest_route.size()))

    @staticmethod
    def aco_medium(gen, no_gen, q, evap):
        maze = Maze.create_maze("./../data/medium maze.txt")
        spec = PathSpecification.read_coordinates("./../data/medium coordinates.txt")
        aco = AntColonyOptimization(maze, gen, no_gen, q, evap)

        start_time = int(round(time.time() * 1000))
        shortest_route = aco.find_shortest_route(spec)
        print("Time taken (medium): " + str((int(round(time.time() * 1000)) - start_time) / 1000.0))
        shortest_route.write_to_file("./../results/medium_solution.txt")
        print("Route size (medium): " + str(shortest_route.size()))

    @staticmethod
    def aco_hard(gen, no_gen, q, evap):
        maze = Maze.create_maze("./../data/hard maze.txt")
        spec = PathSpecification.read_coordinates("./../data/hard coordinates.txt")
        aco = AntColonyOptimization(maze, gen, no_gen, q, evap)

        start_time = int(round(time.time() * 1000))
        shortest_route = aco.find_shortest_route(spec)
        print("Time taken (hard): " + str((int(round(time.time() * 1000)) - start_time) / 1000.0))
        shortest_route.write_to_file("./../results/hard_solution.txt")
        print("Route size (hard): " + str(shortest_route.size()))


# Driver function for Assignment 1
if __name__ == "__main__":
    # parameters
    gen = 100
    no_gen = 100
    q = 10
    evap = 0.8

    # construct the optimization objects
    maze = Maze.create_maze("./../data/medium maze.txt")
    spec = PathSpecification.read_coordinates("./../data/medium coordinates.txt")
    aco = AntColonyOptimization(maze, gen, no_gen, q, evap)

    # save starting time
    start_time = int(round(time.time() * 1000))

    # run optimization
    shortest_route = aco.find_shortest_route(spec)

    # print time taken
    print("Time taken: " + str((int(round(time.time() * 1000)) - start_time) / 1000.0))

    # save solution
    shortest_route.write_to_file("./../results/medium_solution.txt")

    # print route size
    print("Route size: " + str(shortest_route.size()))
