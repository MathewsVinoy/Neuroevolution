import copy
import math
import random

from graphviz import Digraph


class Node:
    def __init__(self, id, type):
        self.id = id
        self.bias = 0
        self.act_fn = None
        self.value = 0
        self.type = type


class Conn:
    _inno_count = 0  # Global innovation number counter

    def __init__(self, links):
        self.inno_no = Conn._inno_count
        Conn._inno_count += 1
        self.links = links  # Tuple: (in_node_id, out_node_id)
        self.weight = random.uniform(-1, 1)
        self.enable = True


class Genome:
    _id = -1

    def __init__(self, nodes, conns):
        self.id = self._set_id()
        self.nodes = nodes  # List of Node instances
        self.conns = conns  # List of Conn instances
        self.fitness = 0

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


class Network:
    def __init__(self, genome):
        self.genome = genome
        self.no_input = 2
        self.no_output = 1

    def activate(self, input_vector):
        for n in self.genome.nodes:
            if n.type == "INPUT":
                n.value = input_vector[n.id]
        print([n.type for n in self.genome.nodes])


class Population:
    def __init__(self):
        self.gen = 0
        self.genomes = []
        self.species = []
        self.create_first()

    def create_first(self):
        input_nodes = [Node(i, "INPUT") for i in range(2)]
        output_nodes = [Node(i + 2, "OUTPUT") for i in range(1)]
        all_nodes = input_nodes + output_nodes

        for _ in range(50):
            conn_links = []
            in_node = random.choice(input_nodes)
            out_node = random.choice(output_nodes)
            conn_links.append(Conn((in_node.id, out_node.id)))

            self.genomes.append(Genome(copy.deepcopy(all_nodes), conn_links))

    def epochs(self):
        print("******** Starting in Gen", self.gen, "********")


def evaluate_fitness(pop):
    INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
    OUTPUTS = [0, 1, 1, 0]
    for g in pop.genomes:
        error = 0.0
        nn = Network(g)
        for i, inputs in enumerate(INPUTS):
            nn.reset()
            output = nn.activate(inputs)
            error += (output[0] - OUTPUTS[i]) ** 2

        g.fitness = 1 - math.sqrt(error / len(OUTPUTS))


if __name__ == "__main__":
    # Create and visualize first genome
    p = Population()
    g = p.genomes[0]
    nn = Network(g)
    nn.activate([1, 0])

    d = Digraph()
    for n in g.nodes:
        label = f"{n.type}_{n.id}"
        d.node(str(n.id), label)

    for c in g.conns:
        if c.enable:
            d.edge(str(c.links[0]), str(c.links[1]), label=f"{c.weight:.2f}")

    d.render("genome_graph", view=True, format="png")  # Saves and opens the file
