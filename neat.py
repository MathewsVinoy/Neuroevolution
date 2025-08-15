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

    def mutate(self):
        if random() < Config.prob_mutatebias:
            self.mutate_bias()

    def mutate_bias(self):
        self.bias += gauss(0, 1) * Config.bias_mutation_power
        if self.bias > Config.max_weight:
            self.bias = Config.max_weight
        elif self.bias < Config.min_weight:
            self.bias = Config.min_weight




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

    def __init__(self, links, weight=None, enable=True):
        self.links = tuple(links)  # (in_node_id, out_node_id)
        self.in_node, self.out_node = self.links
        self.inno_no = Conn.innovation_for(self.links)
        self.weight = random.uniform(-1, 1) if weight is None else float(weight)
        self.enable = bool(enable)

    @classmethod
    def innovation_tracker(cls, links):
        return cls.innovation_for(links)
    
    def mutate(self):
        if random() < Config.prob_mutate_weight:
            self.mutate_weight()
        if random() < Config.prob_togglelink:
            self.enable = True  # Could toggle, but you force-enable here

    def mutate_weight(self):
        self.weight += random.gauss(0, 1) * Config.weight_mutation_power

        if self.weight > Config.max_weight:
            self.weight = Config.max_weight
        elif self.weight < Config.min_weight:
            self.weight = Config.min_weight
        


class Genome:
    _id = -1

    def __init__(self, nodes, conns):
        self.id = self._set_id()
        self.nodes = nodes  # List of Node instances
        self.conns = conns  # List of Conn instances
        self.fitness = 0
        self.specie_id = None

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

    def node_crossover(self, parent1: Node, parent2: Node) -> Node:
        """
        Performs crossover between two nodes to produce a new node.
        """
        assert parent1.id == parent2.id
        bias = random.choice([parent1.bias, parent2.bias])
        act_fun = random.choice([parent1.actfun, parent2.actfun])
        n = Node(idno=parent1.id, ntype=parent1.type)
        n.actfun = act_fun
        n.bias = bias
        return n

    def connection_crossover(
        self, parent1, parent2
    ) :
        """
        Performs crossover between two connections to produce a new connection.
        """
        assert parent1.innoNo == parent2.innoNo
        weight = random.choice([parent1.weight, parent2.weight])
        enable = random.choice([parent1.enable, parent2.enable])
        out = Conn(
            enable=enable,
            links=parent1.links,
            weight=weight,
        )
        out.innoNo = parent1.innoNo
        return out

    def crossover(self, parent2):
        """
        Performs crossover between two genomes to produce a new genome.
        """
        assert self.species_id == parent2.species_id

        conn1 = {conn.innoNo: conn for conn in self.conn}
        conn2 = {conn.innoNo: conn for conn in parent2.conn}
        new_conn = []
        for key, v in conn1.items():
            conn = conn2.get(key)
            if conn != None:
                if key == conn.innoNo:
                    new_conn.append(self.connection_crossover(v, conn))
                else:
                    new_conn.append(v.copy())
            else:
                new_conn.append(v.copy())
        self.conn = new_conn

        node1 = {node.id: node for node in self.nodes}
        node2 = {node.id: node for node in parent2.nodes}
        new_nodes = []
        # print(node2)
        for key, v in node1.items():
            # print(key)
            try:
                node = node2.get(key)
            except KeyError:
                new_nodes.append(v)
            else:
                if node != None:
                    if key == node.id:
                        new_nodes.append(self.node_crossover(v, node))
                    else:
                        new_nodes.append(v)
                else:
                    new_nodes.append(v)

        # print(new_nodes)
        self.next_node_id = max(self.next_node_id, parent2.next_node_id)
        self.nodes = new_nodes


    def mutate(self):
        if random() < 0.03:
            self.addNode()
        elif random() < 0.05:
            self.addConnection()
        else:
            for c in self.conn:
                c.mutate()
            for n in self.nodes:
                if n.type != "INPUT":
                    n.mutate()


class Specie:
    _id = -1

    def __init__(self, rep: Genome):
        self.id = self._set_id()
        self.genomes = []
        self.add(rep)
        self.age = 0
        self.rep = self._get_representative()
        self.hasbest = False
        self.no_improvement_age = 0
        self.spawn_amount =0
        self.last_avg_fitness = 0

    def average_fitness(self):
        if not self.genomes:
            return 0.0  # safe fallback if no genomes exist

        sum_fitness = 0.0
        count = 0
        for g in self.genomes:
            if g.fitness is not None:
                sum_fitness += g.fitness
                count += 1

        if count == 0:
            return 0.0  # all fitness values were None

        current = sum_fitness / count

        if current > self.last_avg_fitness:
            self.last_avg_fitness = current
            self.no_improvement_age = 0
        else:
            self.no_improvement_age += 1

        return current

        

    def _get_representative(self):
        if not self.genomes:
            return None
        return max(self.genomes, key=lambda g: g.fitness)


    @classmethod
    def _set_id(cls):
        cls._id += 1
        return cls._id

    def add(self, g):
        g.specie_id = self.id 
        self.genomes.append(g)

    def reproduce(self):
        offspring = []
        self.age += 1

        assert self.spawn_amount > 0 

        self.genomes.sort(key=lambda g: g.fitness, reverse=True)

        elitism = 1
        if elitism:
            offspring.append(self.genomes[0])
            self.spawn_amount -= 1

        survivors = int(round(len(self.genomes)*0.2))
        if survivors > 0:
            self.genomes = self.genomes[:survivors]
        else:
            self.genomes = self.genomes[:1]

        while(self.spawn_amount> 0):
            self.spawn_amount -=1
            if len(self.genomes) > 1:
                parent1 =  random.choice(self.genomes)
                parent2 = random.choice(self.genomes)
                assert parent1.specie_id == parent2.specie_id
                parent1.crossover(parent2)
                parent1.mutate()
                offspring.append(parent1)
            else:
                parent1 = self.genomes[0]
                parent1.crossover(parent1)
                parent1.mutate()
                offspring.append(parent1)
    
        # print("offspring=",offspring)
        
        self.genomes = []
        self.rep = self._get_representative()

        return offspring
         


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
                    s.add(g)

                    found = True
                    break
            if not found:
                self.species.append(Specie(g))

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

    def compute_spawn_levels(self):
        species_stats =[]
        for s in self.species:
            if s.age < 10:
                species_stats.append(s.average_fitness()*1.2)
            elif s.age > 30:
                species_stats.append(s.average_fitness()*0.2)
            else:
                species_stats.append(s.average_fitness())
        total_average = 0.0 
        for s in species_stats:
            total_average += s

        for i, s in enumerate(self.species):
            s.spawn_amount = int(round((species_stats[i]*50/total_average)))
        

    def epochs(self,size=10):
        for _ in range(size):
            self.gen +=1
            print("******** Starting in Gen", self.gen, "********")
            self.evaluate_fitness(self)
            self.speciate()
            self.best_fitness.append(max(self.genomes, key=lambda g: g.fitness))
            self.avg_fitness.append(self.avg_fitness_fn())
            best = self.best_fitness[-1]

            for s in self.species:
                s.hasbest = False
                if best.specie_id == s.id:
                    s.hasbest = True
            for s in self.species:
                if s.no_improvement_age > 15:
                    if not s.hasbest:
                        for g in self.genomes:
                            if g.specie_id == s.id:
                                self.genomes.remove(g)
                        self.species.remove(s)
                if s.no_improvement_age > 2 * 15:
                    for g in self.genomes:
                        if g.specie_id == s.id:
                            self.genomes.remove(g)
                    self.species.remove(s)
            
            self.compute_spawn_levels()
            for s in self.species:
                if s.spawn_amount == 0:
                    for g in self.genomes[:]:
                        if g.specie_id == s.id:
                            self.genomes.remove(g)
                
            print("no of genomes:=> ", len(self.genomes))

            new_population = []
            for s in self.species:
                new_population = s.reproduce()

            



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

    print("Species id",g.specie_id)
    

    d = Digraph()
    for n in g.nodes:
        label = f"{n.type}_{n.id}"
        d.node(str(n.id), label)

    for c in g.conns:
        print(c.inno_no) 
        if c.enable:
            d.edge(str(c.links[0]), str(c.links[1]), label=f"{c.weight:.2f}")

    d.render("genome_graph", view=True, format="png")  # Saves and opens the file
