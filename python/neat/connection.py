from random import random, gauss
from neat.config import Config

class Connection:
    """
    Connection:
    Represents a connection between two nodes in a NEAT genome.

    - `enable`: boolean, disabled when a new node splits the connection.
    - `innoNo`: unique innovation number per (input_id, output_id) pair.
    """
    
    # Shared innovation tracking across all Connection instances
    innovation_tracker = {}
    innovation_no = 0

    def __init__(self, input_id, out_id, weight, enable):
        self.inId = input_id
        self.outId = out_id
        self.weight = weight
        self.enable = enable
        self.innoNo = self.track_innovation(input_id, out_id)

    @classmethod
    def track_innovation(cls, in_id, out_id):
        key = (in_id, out_id)
        if key not in cls.innovation_tracker.keys():
            cls.innovation_tracker[key] = cls.innovation_no
            cls.innovation_no += 1
        return cls.innovation_tracker[key]

    def mutate(self):
        if random() < Config.prob_mutate_weight:
            self.mutate_weight()
        if random() < Config.prob_togglelink:
            self.enable = True  # Could toggle, but you force-enable here

    def mutate_weight(self):
        self.weight += gauss(0, 1) * Config.weight_mutation_power

        if self.weight > Config.max_weight:
            self.weight = Config.max_weight
        elif self.weight < Config.min_weight:
            self.weight = Config.min_weight

    def copy(self):
        return Connection(self.inId,self.outId, self.weight, self.enable)