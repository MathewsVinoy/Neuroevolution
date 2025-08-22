from neat.activation import exp_activation
from random import random, gauss
from neat.config import Config

class Node:
    def __init__(self, idno: int, ntype: str):
        assert ntype in ('INPUT', 'OUTPUT', 'HIDDEN'), "Invalid node type."
        self.id = idno
        self.type = ntype
        self.actfun = exp_activation
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
        if self.bias > Config.max_weight:
            self.bias = Config.max_weight
        elif self.bias < Config.min_weight:
            self.bias = Config.min_weight

    def mutate_response(self):
       self.response += gauss(0,1)*Config.bias_mutation_power
