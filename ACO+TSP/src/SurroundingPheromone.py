import os
import sys

from Direction import Direction

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


# Class containing the pheromone information around a certain point in the maze
class SurroundingPheromone:
    # Creates a surrounding pheromone object.
    # @param north the amount of pheromone in the north.
    # @param east the amount of pheromone in the east.
    # @param south the amount of pheromone in the south.
    # @param west the amount of pheromone in the west.
    def __init__(self, north, east, south, west):
        self.dirs = [east, north, west, south]
        self.total_surrounding_pheromone = east + north + west + south

    # Get the total amount of surrounding pheromone.
    # @return total surrounding pheromone
    def get_total_surrounding_pheromone(self):
        return self.total_surrounding_pheromone

    # Get a specific pheromone level
    # @param dir Direction of pheromone
    # @return Pheromone of dir
    def get(self, dir):
        return self.dirs[Direction.dir_to_int(dir)]

    def __str__(self):
        return "North: " + str(self.dirs[1]) + " East: " + str(self.dirs[0]) \
               + " South: " + str(self.dirs[3]) + " West: " + str(self.dirs[2])
