from neat.activation import sigmoid_activation
from random import random, gauss
from neat.config import Config

class Node:
    def __init__(self, idno: int, ntype: str):
        assert ntype in ('INPUT', 'OUTPUT', 'HIDDEN'), "Invalid node type."
        self.id = idno
        self.type = ntype
        self.actfun = sigmoid_activation
        self.bias = 0.0
        self.output = 0.0
        self.response = 4.924273  # default NEAT response gain

    def mutate(self):
        if random() < Config.prob_mutatebias:
            self.mutate_bias()
        if random() < Config.prob_mutatebias:
            self.mutate_response()

    def mutate_bias(self):
        self.bias += gauss(0, 1) * Config.bias_mutation_power
        self.bias = max(Config.min_weight, min(self.bias, Config.max_weight))

    def mutate_response(self):
        self.response += gauss(0, 1) * Config.bias_mutation_power
        self.response = max(Config.min_weight, min(self.response, Config.max_weight))
