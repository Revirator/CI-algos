import os
import sys
import time

from AntColonyOptimization import AntColonyOptimization
from Maze import Maze
from PathSpecification import PathSpecification
from TSPData import TSPData
from GeneticAlgorithm import GeneticAlgorithm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


if __name__ == "__main__":
    gen_easy = 25
    no_gen_easy = 10
    q_easy = 100

    gen_medium = 100
    no_gen_medium = 50
    q_medium = 200

    gen_hard = 120
    no_gen_hard = 60
    q_hard = 400
    evap = 0.8

    # EASY MAZE
    AntColonyOptimization.aco_easy(gen_easy, no_gen_easy, q_easy, evap)

    # MEDIUM MAZE
    AntColonyOptimization.aco_medium(gen_medium, no_gen_medium, q_medium, evap)

    # HARD MAZE
    AntColonyOptimization.aco_hard(gen_hard, no_gen_hard, q_hard, evap)

    # TSP PART

    # parameters
    population_size = 500
    generations = 5000
    mutation_prob = 0.02
    crossover_prob = 0.7
    persistFile = "./../tmp/productMatrixDist"

    # setup optimization
    # The distance matrix can be generated using the TSPData.py, but it might take a few hours
    tsp_data = TSPData.read_from_file(persistFile)
    ga = GeneticAlgorithm(generations, population_size, crossover_prob, mutation_prob)
    solution = ga.solve_tsp(tsp_data)
    tsp_data.write_action_file(solution, "./../data/TSP solution.txt")



