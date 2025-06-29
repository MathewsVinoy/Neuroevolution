from neat.activation import sigmoid_activation
from random import random, gauss
from neat.config import Config

class Node:
    def __init__(self,idno:int, ntype: str):
        self.id = idno
        self.type = ntype
        self.actfun = sigmoid_activation
        self.bias = 0
        self.output =0.0
        assert(self._type in ('INPUT', 'OUTPUT', 'HIDDEN'))
    
    def mutate(self):
        if random() < Config.prob_mutatebias:
            self.mutate_bias()
        if random() < Config.prob_mutatebias:
            self.mutate_response()
    

    def mutate_bias(self):
        self._bias += random.gauss(0,1)*Config.bias_mutation_power
        if self._bias > Config.max_weight:
            self._bias = Config.max_weight
        elif self._bias < Config.min_weight:
            self._bias = Config.min_weight

    def mutate_response(self):
        self._response += random.gauss(0,1)*Config.bias_mutation_power