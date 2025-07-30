from neat import config
from neat.population import Population
from neat.network import Network
import math

config.load('xor2_config')

INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
OUTPUTS = [0, 1, 1, 0]

def eval_function(pop: Population):
    for g in pop.genomes:
        error = 0.0
        nn = Network(g)
        for i, inputs in enumerate(INPUTS):
            nn.reset()
            output = nn.activate(inputs)
            error += (output[0]-OUTPUTS[i])**2

        g.fitness = 1-math.sqrt(error/len(OUTPUTS))

Population.evaluate_Fitness = eval_function

Population().evolve(no=20)

