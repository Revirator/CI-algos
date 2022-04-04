from locale import currency
import os
import random
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from TSPData import TSPData

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


# TSP problem solver using genetic algorithms.
class GeneticAlgorithm:


    # Constructs a new 'genetic algorithm' object.
    # @param generations the amount of generations.
    # @param popSize the population size.
    def __init__(self, generations, pop_size, crossover_prob, mutation_prob):
        self.generations = generations
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    # Knuth-Yates shuffle, reordering a array randomly
    # @param chromosome array to shuffle.
    def shuffle(self, chromosome):
        n = len(chromosome)
        for i in range(n):
            r = i + int(random.uniform(0, 1) * (n - i))
            swap = chromosome[r]
            chromosome[r] = chromosome[i]
            chromosome[i] = swap
        return chromosome

    # This method should solve the TSP.
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data):
        start_dist = tsp_data.get_start_distances()
        end_dist = tsp_data.get_end_distances()
        product_dist_matrix = tsp_data.get_distances()

        products = [i for i in range(len(start_dist))]
        
        current_population = []

        # Initial population
        for p in range(self.pop_size):
            chromosome = shuffle(products)
            current_population.append(chromosome)

        distances = []
        for g in range(self.generations):
            new_population = []

            # print("Distance sum for generation " + str(g) + ":")
            # Calculate the fitness ratio for each chromosome in the current population
            fitness_ratio, dist_sum = self.calculate_fitness_ratio(current_population, start_dist, end_dist, product_dist_matrix)
            distances.append(dist_sum)
            # Choose parent chromosomes for the new population based on their fitness ratios
            chosen_chromosome_indices = np.random.choice(len(current_population), len(current_population), replace=True, p=fitness_ratio)
            chosen_chromosomes = [current_population[idx] for idx in chosen_chromosome_indices]

            # With every 2 parents, create 2 offsprings via crossover and mutation
            for i in range(len(chosen_chromosomes) // 2):
                # Crossover
                offspringA, offspringB = self.crossover(chosen_chromosomes[2*i], chosen_chromosomes[2*i + 1])
                # Mutation
                offspringA = self.mutate(offspringA)
                offspringB = self.mutate(offspringB)

                new_population.append(offspringA)
                new_population.append(offspringB)
            


            current_population = new_population
        
        # plt.plot(range(1, self.generations+1), distances)
        # plt.xticks(range(1, self.generations+1))
        # plt.xlabel("Generation")
        # plt.ylabel("Sum of distances")
        # plt.show()

        fitness_ratio, a = self.calculate_fitness_ratio(current_population, start_dist, end_dist, product_dist_matrix)
        return current_population[np.argmax(fitness_ratio)]

    def crossover(self, chrom_l, chrom_r):
        if np.random.uniform() < self.crossover_prob:
            # Select random start and end indices for the start and end of a subsequence which will be used preserved in the crossover
            left = int(np.random.uniform() * len(chrom_l))
            right = int(np.random.uniform() * len(chrom_l))
            left = right
            if left > right:
                left, right = right, left

            preserved_l, preserved_r = chrom_l[left:right + 1], chrom_r[left:right + 1]
            offspring_l, offspring_r = [], []
            curr_l_idx, curr_r_idx = 0, 0

            for i in range(len(chrom_r)):
                # If index is within the preserved range - copy the preserved subsequence into the offspring
                if left <= i <= right:
                    offspring_l.append(preserved_l[i-left])
                    offspring_r.append(preserved_r[i-left])
                # Else find the first non included product in the other parent and copy it
                else:
                    while chrom_r[curr_r_idx] in preserved_l:
                        curr_r_idx += 1
                    while chrom_l[curr_l_idx] in preserved_r:
                        curr_l_idx += 1
                    offspring_l.append(chrom_r[curr_r_idx])
                    curr_r_idx += 1
                    offspring_r.append(chrom_l[curr_l_idx])
                    curr_l_idx += 1

            return offspring_l, offspring_r
        else:
            return chrom_l, chrom_r


    def mutate(self, chromosome):
        if np.random.uniform() < self.mutation_prob:
            # Swap any two random nodes to mutate the path
            # 2 3 1 4 5 7 6
            # 2 3 7 4 5 1 6
            left = int(np.random.uniform() * len(chromosome))
            right = int(np.random.uniform() * len(chromosome))
            chromosome[left], chromosome[right] = chromosome[right], chromosome[left]

            return chromosome
        else:
            return chromosome

    # Fitness as the total distance of the
    def calculate_fitness_ratio(self, population, start_dist, end_dist, dist_matrix):
        fitness_arr = []
        dist_sum = 0.0
        fitness_sum = 0.0
        
        # Calculate the fitness for each chromosome
        for chromosome in population:
            # Add the path from the start to the first product
            res = start_dist[chromosome[0]]
            for i in range(len(chromosome)-1):
                res += dist_matrix[chromosome[i]][chromosome[i+1]]
            # Add the path from the last product to the end
            res += end_dist[chromosome[-1]]
            fitness_arr.append(res)
            dist_sum += float(res)
        
        # print(dist_sum)
        # Calculate the fitness ratio for each chromosome
        for i in range(len(fitness_arr)):
            fitness_arr[i] = dist_sum / float(fitness_arr[i])
            fitness_sum += fitness_arr[i]

        ratio_sum = 0.0
        for i in range(len(fitness_arr)):
            fitness_arr[i] = fitness_arr[i] / fitness_sum
            ratio_sum += fitness_arr[i]

        fitness_arr[-1] += (1 - ratio_sum)

        return fitness_arr, dist_sum

# Assignment 2.b
if __name__ == "__main__":
    # parameters
    population_size = 500
    generations = 5000
    mutation_prob = 0.02
    crossover_prob = 0.7
    persistFile = "./../tmp/productMatrixDist"

    # setup optimization
    tsp_data = TSPData.read_from_file(persistFile)
    ga = GeneticAlgorithm(generations, population_size, crossover_prob, mutation_prob)
    solution = ga.solve_tsp(tsp_data)
    tsp_data.write_action_file(solution, "./../data/TSP solution.txt")