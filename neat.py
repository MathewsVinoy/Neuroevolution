import copy
import math
import random

from graphviz import Digraph

def sigmoid_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))


class Node:
    def __init__(self, id, type):
        self.id = id
        self.bias = 0
        self.act_fn = sigmoid_activation
        self.value = 0
        self.type = type



class Conn:
    _innovation_tracker = {}
    _inno_count = 0

    @classmethod
    def innovation_for(cls, links):
        links = tuple(links)
        if len(links) != 2:
            raise ValueError(f"links must be a (in_id, out_id) pair, got: {links}")
        if links not in cls._innovation_tracker:
            cls._innovation_tracker[links] = cls._inno_count
            cls._inno_count += 1
        return cls._innovation_tracker[links]

    def __init__(self, links, weight=None, enable=True, inno_no=None):
        self.links = tuple(links)  # (in_node_id, out_node_id)
        self.in_node, self.out_node = self.links
        self.inno_no = Conn.innovation_for(self.links)
        self.weight = random.uniform(-1, 1) if weight is None else float(weight)
        self.enable = bool(enable)

    @classmethod
    def innovation_tracker(cls, links):
        return cls.innovation_for(links)
        


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
    
    def distance(self,other):
        if len(self.conns) > len(other.conns):
            parent1 = self.conns
            parent2 = other.conns
        else:
            parent1 = other.conns
            parent2 = self.conns

        matching = 0
        execess =0
        disjoint =0
        weight_diff =0.0

        max_cg_parent2 = max(c.inno_no for c in parent2)
        cg2 = {c.inno_no: c for c in parent2}
        for c1 in parent1:
            try:
                c2 = cg2[c1.inno_no]
            except KeyError:
                if c1.inno_no > max_cg_parent2:
                    execess += 1
                else:
                    disjoint += 1
            else:
                weight_diff += math.fabs(c1.weight - c2.weight)
                matching += 1
        disjoint += len(parent2) - matching
        distance = (
            0.1 * execess +
            0.1 * disjoint
        )
        if matching > 0:
            distance += 0.4 * (weight_diff / matching)

        return distance



class Specie:
    _id = -1

    def __init__(self, genomes):
        self.id = self._set_id()
        self.genomes = genomes
        self.rep = self._get_representative()

    def _get_representative(self):
        if not self.genomes:
            return None
        return max(self.genomes, key=lambda g: g.fitness)

    @classmethod
    def _set_id(cls):
        cls._id += 1
        return cls._id


class Network:
    def __init__(self, genome):
        self.genome = genome
        self.no_input = 2
        self.no_output = 1
    
    def reset(self):
        for node in self.genome.nodes:
            node.value = 0.0

    def activate(self, input_vector):
        # Reset node values
        for node in self.genome.nodes:
            node.value = 0

        # Set input values
        for i, value in enumerate(input_vector):
            self.genome.nodes[i].value = value

        # Activate connections
        for conn in self.genome.conns:
            if conn.enable:
                in_node = next(n for n in self.genome.nodes if n.id == conn.links[0])
                out_node = next(n for n in self.genome.nodes if n.id == conn.links[1])
                out_node.value += in_node.act_fn(in_node.value) * conn.weight

        # Return output values
        return [n.value for n in self.genome.nodes if n.type == "OUTPUT"]


class Population:
    evaluate_fitness = None
    _compatibility_threshold = 3.0
    def __init__(self):
        self.gen = 0
        self.genomes = []
        self.species = []
        self.best_fitness = []
        self.avg_fitness =[]
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

    def speciate(self):
        for g  in self.genomes:
            found = False
            for s in self.species:
                if g.distance(s.rep) < self._compatibility_threshold:
                    s.genomes.append(g)
                    found = True
                    break
            if not found:
                self.species.append(Specie([g]))

        

        self._set_compatibility_threshold()

    def _set_compatibility_threshold(self):
        if len(self.species) > 10:
            self._compatibility_threshold += 0.1
        elif len(self.species) < 10:
            if self._compatibility_threshold > 0.1:
                self._compatibility_threshold -= 0.1
    
    def avg_fitness_fn(self):
        sum = 0.0
        for g in self.genomes:
            sum += g.fitness
        return sum/len(self.genomes)
        

    def epochs(self,size=10):
        for _ in range(size):
            self.gen +=1
            print("******** Starting in Gen", self.gen, "********")
            self.evaluate_fitness(self)
            self.speciate()
            self.best_fitness.append(max(self.genomes, key=lambda g: g.fitness))
            self.avg_fitness.append(self.avg_fitness_fn())
            best = self.best_fitness[-1]
            



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
    p.evaluate_fitness = evaluate_fitness
    p.epochs(2)
    print("Number of species:", len(p.species))
    g = p.best_fitness[-1]
    print(p.best_fitness)
    print(p.avg_fitness)
    nn = Network(g)
    out =nn.activate([1, 0])
    print("OutPut:",out)
    

    d = Digraph()
    for n in g.nodes:
        label = f"{n.type}_{n.id}"
        d.node(str(n.id), label)

    for c in g.conns:
        print(c.inno_no) 
        if c.enable:
            d.edge(str(c.links[0]), str(c.links[1]), label=f"{c.weight:.2f}")

    d.render("genome_graph", view=True, format="png")  # Saves and opens the file
