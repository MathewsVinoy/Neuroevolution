import random


class Node:
    def __init__(self, id, type):
        self.id = id
        self.bias = 0
        self.act_fn = None
        self.value = 0
        self.type = type


class Conn:
    def __init__(self, links):
        self.inno_no = 0
        self.links = links
        self.weight = 0
        self.enable = True


class Genome:
    _id = -1

    def __init__(self, nodes, conns):
        self.id = self._set_id()
        self.nodes = nodes
        self.conns = conns

    @classmethod
    def _set_id(cls):
        cls._id += 1
        return cls._id


class Specie:
    _id = -1

    def __init__(self, genomes):
        self.id = self._set_id()
        self.genomes = genomes

    @classmethod
    def _set_id(cls):
        cls._id += 1
        return cls._id


class Population:
    def __init__(self):
        self.gen = 0
        self.genomes = []
        self.species = []
        self.create_first()

    def create_first(self):
        n1 = []
        for i in range(2):
            n1.append(Node(i, "INPUT"))
        n2 = []
        for i in range(1):
            n2.append(Node(2 + i, "OUTPUT"))
        for i in range(50):
            link = (random.choice(n1), random.choice(n2))
            self.genomes.append(Genome(n1.extend(n2), [Conn(link)]))

    def epochs(self):
        print("******** Starting in Gen ", self.gen, " ********")


p = Population()
g = p.genome[0]
