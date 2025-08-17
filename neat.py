import copy
import math
import random

from graphviz import Digraph

def exp_activation(z):
    z = max(-60.0, min(60.0, z))
    return math.exp(z)


class Node:
    def __init__(self, id, type):
        self.id = id
        self.bias = 0
        self.act_fn = exp_activation
        self.value = 0
        self.type = type

    def mutate(self):
        if random.random() < 0.20:
            self.mutate_bias()

    def mutate_bias(self):
        self.bias += random.gauss(0, 1) * 0.50
        if self.bias > 30:
            self.bias = 30
        elif self.bias < -30:
            self.bias = -30




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
        if random.random() < 0.90:
            self.mutate_weight()
        if random.random() < 0.01:
            self.enable = True  # Could toggle, but you force-enable here

    def mutate_weight(self):
        self.weight += random.gauss(0, 1) * 0.50

        if self.weight > 30:
            self.weight = 30
        elif self.weight <-30:
            self.weight =-30

    def copy(self):
        return Conn(self.links,self.weight,self.enable)
        


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
        act_fun = random.choice([parent1.act_fn, parent2.act_fn])
        n = Node(parent1.id,parent1.type)
        n.act_fn = act_fun
        n.bias = bias
        return n

    def connection_crossover(
        self, parent1, parent2
    ) :
        """
        Performs crossover between two connections to produce a new connection.
        """
        assert parent1.inno_no == parent2.inno_no
        weight = random.choice([parent1.weight, parent2.weight])
        enable = random.choice([parent1.enable, parent2.enable])
        out = Conn(
            enable=enable,
            links=parent1.links,
            weight=weight,
        )
        out.inno_no = parent1.inno_no
        return out

    def crossover(self, parent2):
        """
        Performs crossover between two genomes to produce a new genome.
        """
        assert self.specie_id == parent2.specie_id

        conn1 = {conn.inno_no: conn for conn in self.conns}
        conn2 = {conn.inno_no: conn for conn in parent2.conns}
        new_conns = []
        for key, v in conn1.items():
            c2 = conn2.get(key)
            if c2 is not None and key == c2.inno_no:
                new_conns.append(self.connection_crossover(v, c2))
            else:
                new_conns.append(v.copy())
        self.conns = new_conns

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

        self.nodes = new_nodes


    def mutate(self):
        if random.random() < 0.03:
            self.addNode()
        elif random.random() < 0.05:
            self.addConnection()
        else:
            for c in self.conns:
                c.mutate()
            for n in self.nodes:
                if n.type != "INPUT":
                    n.mutate()
    
    def addNode(self):
        if not self.conns:
            return
        old_conn = random.choice(self.conns)
        # Create and register the new hidden node
        n1 = Node(len(self.nodes), "HIDDEN")
        self.nodes.append(n1)

        # Split the old connection
        c1 = Conn(
            links=(old_conn.links[0], n1.id),
            weight=1,
            enable=True,
        )
        c2 = Conn(
            links=(n1.id, old_conn.links[1]),
            weight=old_conn.weight,
            enable=True,
        )
        old_conn.enable = False
        # Replace old connection with the two new ones
        self.conns.remove(old_conn)
        self.conns.append(c1)
        self.conns.append(c2)

    def addConnection(self):
        n1, n2 = random.sample(self.nodes, 2)
        if n1.id == n2.id:
            return

        # Disallow connections between two inputs or two outputs
        if n1.type == "INPUT" and n2.type == "INPUT":
            return
        if n1.type == "OUTPUT" and n2.type == "OUTPUT":
            return

        # Decide direction so that source is not OUTPUT and target is not INPUT
        if n1.type == "INPUT" or n2.type == "OUTPUT":
            source, target = n1, n2
        elif n2.type == "INPUT" or n1.type == "OUTPUT":
            source, target = n2, n1
        else:
            # Both are HIDDEN or one is HIDDEN and the other is INPUT/OUTPUT already handled.
            # Randomize direction for hidden-hidden pairs but avoid creating an INPUT target or OUTPUT source
            if random.random() < 0.5:
                source, target = n1, n2
            else:
                source, target = n2, n1

        # Prevent self-connection
        if source.id == target.id:
            return

        # Avoid duplicate
        if any(c.links == (source.id, target.id) for c in self.conns):
            return

        # Avoid cycles: check if adding source->target would create a cycle
        if self.check_cycle((source, target)):
            return

        self.conns.append(Conn(
            links=(source.id, target.id),
            enable=True,
            weight=random.uniform(-1, 1),
        ))
    
    def check_cycle(self, node):
        # node is a tuple (source_node, target_node)
        def dfs(current_id, target_id, visited, conns):
            if current_id == target_id:
                return True
            visited.add(current_id)
            for conn in conns:
                if conn.enable and conn.links[0] == current_id:
                    nxt = conn.links[1]
                    if nxt not in visited and dfs(nxt, target_id, visited, conns):
                        return True
            return False

        source = node[0].id
        target = node[1].id
        return dfs(target, source, set(), self.conns)


class Specie:
    _id = -1

    def __init__(self, rep: Genome):
        self.id = self._set_id()
        self.genomes = []
        self.rep = rep  # keep given representative
        self.age = 0
        self.hasbest = False
        self.no_improvement_age = 0
        self.spawn_amount = 0
        self.last_avg_fitness = 0
        if rep is not None:
            self.add(rep)

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

        # Sort by fitness (descending)
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)
        best_parent = self.genomes[0] if self.genomes else None

        # Elitism
        elitism = 1
        if elitism and best_parent is not None:
            offspring.append(copy.deepcopy(best_parent))
            self.spawn_amount -= 1

        # Truncate to survivors
        survivors = int(round(len(self.genomes) * 0.2))
        if survivors > 0:
            self.genomes = self.genomes[:survivors]
        else:
            self.genomes = self.genomes[:1]

        # Produce children
        while self.spawn_amount > 0:
            self.spawn_amount -= 1
            if len(self.genomes) > 1:
                parent1 = random.choice(self.genomes)
                parent2 = random.choice(self.genomes)
                assert parent1.specie_id == parent2.specie_id
                child = copy.deepcopy(parent1)
                child.crossover(parent2)
                child.mutate()
                offspring.append(child)
            else:
                parent = self.genomes[0]
                child = copy.deepcopy(parent)
                child.crossover(parent)
                child.mutate()
                offspring.append(child)

        # Keep representative stable across generations (use best parent)
        if best_parent is not None:
            self.rep = best_parent

        # Clear members; they will be reassigned by Population.speciate()
        self.genomes = []

        return offspring
         


class Network:
    """
    Network (Phenotype):
    Represents the expressed neural network constructed from a genome.
    """

    def __init__(self, genome: Genome):
        self.nodes = genome.nodes
        self.conn = [c for c in genome.conns if c.enable]  # Only enabled connections

    def reset(self):
        for n in self.nodes:
            n.output = 0.0

    def activate(self, input_vector):
        assert len(input_vector) == 2, "Wrong number of inputs."

        # Set input node outputs
        input_index = 0
        for n in self.nodes:
            if n.type == 'INPUT':
                n.output = input_vector[input_index]
                input_index += 1

        # Sort nodes in topological order if not already done
        sorted_nodes = sorted(self.nodes, key=lambda x: x.type)

        # Process non-input nodes
        for node in sorted_nodes:
            if node.type == 'INPUT':
                continue

            total_input = 0.0
            for c in self.conn:
                if c.links[1] == node.id:
                    input_node = next((n for n in self.nodes if n.id == c.links[0]), None)
                    if input_node is not None:
                        total_input += c.weight * input_node.output

            node.output = node.act_fn(total_input)

        # Collect outputs
        output_values = [n.output for n in self.nodes if n.type == 'OUTPUT']
        return output_values


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
            conns = []
            in_node = random.choice(input_nodes)
            out_node = random.choice(output_nodes)
            conns.append(Conn((in_node.id, out_node.id)))
            self.genomes.append(Genome(copy.deepcopy(all_nodes), conns))

    def speciate(self):
        # If no species exist, seed with first genome and grow
        if not self.species:
            for g in self.genomes:
                placed = False
                for s in self.species:
                    if s.rep is not None and g.distance(s.rep) < self._compatibility_threshold:
                        s.add(g)
                        placed = True
                        break
                if not placed:
                    self.species.append(Specie(g))
        else:
            # Keep representatives but clear memberships
            for s in self.species:
                s.genomes = []

            # Reassign all genomes
            for g in self.genomes:
                placed = False
                for s in self.species:
                    if s.rep is not None and g.distance(s.rep) < self._compatibility_threshold:
                        s.add(g)
                        placed = True
                        break
                if not placed:
                    self.species.append(Specie(g))

            # Remove empty species
            self.species = [s for s in self.species if s.genomes]

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
        species_stats = []
        for s in self.species:
            if s.age < 10:
                species_stats.append(s.average_fitness() * 1.2)
            elif s.age > 30:
                species_stats.append(s.average_fitness() * 0.2)
            else:
                species_stats.append(s.average_fitness())
        total_average = sum(species_stats) if species_stats else 0.0
        if total_average == 0.0:
            # Split evenly to maintain population size 50
            even = int(round(50 / max(1, len(self.species))))
            for s in self.species:
                s.spawn_amount = even
            return

        for i, s in enumerate(self.species):
            s.spawn_amount = int(round((species_stats[i] * 50 / total_average)))
        

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
                s.hasbest = (best.specie_id == s.id)

            for s in self.species[:]:
                if s.no_improvement_age > 15 and not s.hasbest:
                    # remove species and its genomes
                    self.genomes = [g for g in self.genomes if g.specie_id != s.id]
                    self.species.remove(s)
                elif s.no_improvement_age > 30:
                    self.genomes = [g for g in self.genomes if g.specie_id != s.id]
                    self.species.remove(s)

            self.compute_spawn_levels()

            # Remove genomes from species that won't spawn
            for s in self.species:
                if s.spawn_amount == 0:
                    self.genomes = [g for g in self.genomes if g.specie_id != s.id]

            # Remove species with zero spawn amounts
            self.species = [s for s in self.species if s.spawn_amount > 0]

            print("no of genomes:=> ", len(self.genomes))

            # Reproduce to build the next population
            new_population = []
            for s in self.species:
                new_population.extend(s.reproduce())

            fill = 50 - len(new_population)
            if fill < 0:
                new_population = new_population[:50]
            if fill > 0:
                while fill > 0:
                    parent1 = random.choice(self.genomes)
                    found = False
                    for c in self.genomes:
                        if c.specie_id == parent1.specie_id:
                            parent1.crossover(c)
                            parent1.mutate()
                            new_population.append(parent1)
                            found = True
                            break
                    if not found:
                        parent1.mutate()
                        new_population.append(parent1)
                    fill -= 1
            print(f"Species: {len(self.species)}, Population: {len(new_population)}")  # Debugging statement
            assert 50 == len(new_population), 'Different population sizes!'
            self.species =[]
            self.genomes = new_population



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

def test(best):
    print("\n\n\n############# Test ##########")
    print("Best genome fitness:", best.fitness)
    nn = Network(best)

    while True:
        s = input("Enter two inputs (e.g. '0 1'), 'all' to test XOR cases, or 'q' to quit: ").strip()
        if not s:
            continue
        if s.lower() in ("q", "quit", "exit"):
            print("Exiting test.")
            break

        if s.lower() == "all":
            cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
            for c in cases:
                nn.reset()
                out = nn.activate(c)
                print(f"{c} -> {out}")
            continue

        s = s.replace(",", " ")
        parts = s.split()
        if len(parts) != 2:
            print("Please enter two values separated by space or comma.")
            continue
        try:
            vals = [float(parts[0]), float(parts[1])]
        except ValueError:
            print("Invalid numeric values.")
            continue

        nn.reset()
        out = nn.activate(vals)
        print(f"Input: {vals} -> Output: {out}")



if __name__ == "__main__":
    # Create and visualize first genome
    p = Population()
    p.evaluate_fitness = evaluate_fitness
    p.epochs(50)
    print("\n\n\n\n\n\n\n\n\n\n###################Debug Report###################")
    print("Number of species:", len(p.species))
    g = p.best_fitness[-1]
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

    d.render("genome_graph", view=True, format="png") 
    test(g)
