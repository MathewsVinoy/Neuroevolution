import random
from activation import sigmoid_activation


innovation_numbers=[]
node_numbers=[]
innovation_no = 0

node_no =0

compatibility_threshold = 3.0
POPULATIONA_SIZE =1

class Node:
    def __init__(self,idno:int, ntype: str):
        self.id = idno
        self.type = ntype
        self.actfun = sigmoid_activation
        self.bias = 0

class Connection:
    """
    Connection:
     This class is used to creat an istance of the connection between 2 nodes.

     self.enable => (dtype = boolen) diable when new node is added then added to the last of the Genome
     innovation numbers =>(dtype = int) unique number reprsinting each connection
    """
    
    def __init__(self,input_id,out_id,weight,enable,innovation_number):
        self.inId = input_id
        self.outId = out_id
        self.weight = weight
        self.enable = enable  
        self.innoNo = innovation_number 

class Genome:
    """
        Genotype(Genome):
            things the are not visible
    """
    def __init__(self,node:list[Node], connection:list[Connection]):
        self.nodes = node
        self.conn = connection
        self.fitness = 0


    # mutation add connection
    def addConnection(self):
        if len(self.nodes) < 2:
            raise RuntimeError("Need 2 or more than 2 nodes")

        # Create a graph representation of the current connections
        graph = {node.id: [] for node in self.nodes}
        for conn in self.conn:
            if conn.enable:
                graph[conn.inId].append(conn.outId)

        # Try to add a new connection
        for _ in range(len(self.nodes) * (len(self.nodes) - 1)):
            n1, n2 = random.sample(self.nodes, 2)
            if not self._creates_cycle(graph, n1.id, n2.id):
                weight = random.uniform(-1, 1)
                inn_no = len(innovation_numbers)
                innovation_numbers.append(inn_no)
                new_connection = Connection(
                    input_id=n1.id,
                    out_id=n2.id,
                    enable=True,
                    innovation_number=inn_no,
                    weight=weight
                )
                self.conn.append(new_connection)
                graph[n1.id].append(n2.id)  # Update the graph with the new connection
                break

    def _creates_cycle(self, graph, start, end):
        visited = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node == end:
                return True
            if node not in visited:
                visited.add(node)
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
        return False

    
    # mutation add Node
    def addNode(self):
        old_conn = random.choice(self.conn)
        old_conn.enable=False
        no = len(node_numbers)
        node_numbers.append(no)
        new_node = Node(
            ntype=type,
            idno=innovation_no,
        )
        inn_no = len(innovation_numbers)
        innovation_numbers.append(inn_no)
        conn1 = Connection(
            input_id=old_conn.inId,
            enable=True,
            innovation_number=inn_no,
            out_id= new_node.id,
            weight=1,
        )
        inn_no = len(innovation_numbers)
        innovation_numbers.append(inn_no)
        conn2 = Connection(
            input_id=new_node.id,
            enable=True,
            innovation_number=inn_no,
            out_id= old_conn.outId,
            weight=old_conn.weight,
        )
    
    def node_crossover(self, parent1:Node,parent2:Node)->Node:
        assert parent1.id==parent2.id
        bias = random.choice([parent1.bias,parent2.bias])
        act_fun = random.choice([parent1.actfun,parent2.actfun])
        n =Node(idno=parent1.id,ntype=parent1.type)
        n.actfun = act_fun
        n.bias = bias
        return n
            

    def connection_crossover(self,parent1: Connection,parent2:Connection)->Connection:
        assert parent1.innoNo == parent2.innoNo
        weight = random.choice([parent1.weight,parent2.weight])
        enable = random.choice([parent1.enable,parent2.enable])
        return Connection(innovation_number=parent1.innoNo,input_id=parent1.inId,out_id=parent1.outId,weight=weight,enable=enable)



    def Crossover(self,parent2):
        conn_dict1 = {conn.innoNo: conn for conn in self.conn}
        conn_dict2 = {conn.innoNo: conn for conn in parent2.conn}

        all_conn = sorted(set(conn_dict1.keys()) | set(conn_dict2.keys()))


        if self.fitness > parent2.fitness:
            fitness = self.fitness
        else:
            fitness = parent2.fitness


        new_conn = []
        for i in all_conn:
            conn1 = conn_dict1.get(i)
            conn2 = conn_dict2.get(i)
            print(conn1)

            if conn1 and conn2:
                new_conn.append(self.connection_crossover(conn1, conn2))
            elif conn1:
                new_conn.append(conn1)
            elif conn2:
                new_conn.append(conn2)

        self.conn = new_conn
        new_nodes =[]
        for n1 in self.nodes:
            for n2 in parent2.nodes:
                if n1.id == n2.id:
                    new_nodes.append(self.node_crossover(n1,n2))
        
        self.node = new_nodes

    

class Network:
    """
    Network (Phenotype):
    Changes that can be seen / the changes that are visible.
    """
    def __init__(self, genome: Genome):
        self.node = genome.nodes
        self.conn = genome.conn
        self.node = self._topological_sort()
    
    

    def _topological_sort(self):
        in_degree = {node.id: 0 for node in self.node}
        for conn in self.conn:
            if conn.enable:
                in_degree[conn.outId] += 1

        queue = [node for node in self.node if in_degree[node.id] == 0]

        topo_order = []

        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for conn in self.conn:
                if conn.inId == node.id and conn.enable:
                    in_degree[conn.outId] -= 1
                    if in_degree[conn.outId] == 0:
                        neighbor_node = next(n for n in self.node if n.id == conn.outId)
                        queue.append(neighbor_node)

        if len(topo_order) != len(self.node):
            raise ValueError("The network contains a cycle and cannot be topologically sorted.")

        return topo_order

    def activate(self, input_vector):
        input_nodes = [node for node in self.node if node.type == 'input']
        output_nodes = [node for node in self.node if node.type == 'output']

        if len(input_nodes) != len(input_vector):
            raise RuntimeError("The number of inputs and input nodes does not match.")

        values = {node.id: 0.0 for node in self.node}

        for node, value in zip(input_nodes, input_vector):
            values[node.id] = value

        for node in self.node:
            total = 0.0
            for conn in self.conn:
                if conn.outId == node.id and conn.enable:
                    total += values[conn.inId] * conn.weight
            if node.type != 'input':
                values[node.id] = node.actfun(total + node.bias)
        # print([values[node.id] for node in output_nodes])
        return [values[node.id] for node in output_nodes]

    def _build_network_graph(self, genome:Genome):
        pass
            



class Specie:
    """
    species is represented by a random genome inside the species from the previous generation
    """

    def __init__(self,rep: Genome):
        self.genomes= []
        self.id = 0
        self.rep = rep
        self.staleness = 0.0
        self.fitness =0.0

    def compatibilityDistance(self, genome: Genome,c1=1.0, c2=1.0, c3=0.4):
        inGenome = {conn.innoNo: conn for conn in genome.conn}
        repGenome = {conn.innoNo: conn for conn in self.rep.conn}

        innovationNo1 = set(inGenome.keys())
        innovationNo2 = set(repGenome.keys())

        match_genes =  innovationNo1 & innovationNo2
        disjoint_genes = (innovationNo1^innovationNo2)-(innovationNo1|innovationNo2).difference(match_genes)
        exec_genes = {
            inn for inn in innovationNo1 | innovationNo2 
            if(inn> max(innovationNo1)and inn in innovationNo2)or 
            (inn >max(innovationNo2)and inn in innovationNo1)
        }
        if match_genes:
            W= sum(abs(inGenome[i].weight-repGenome[i].weight)for i in match_genes)
        else:
            W=0.0

        N= max(len(genome.conn),len(self.rep.conn))
        E = len(exec_genes)
        D = len(disjoint_genes)

        delta = (c1*E)/N + (c2*D)/N +(c3* W)
        return delta

class Population:
    def __init__(self):
        self.species = []
        self.current_generation = 0
        self.genomes = []
        self.node_no =0
        self.innovation_no = 0
    
    def remove_members(self, species: Specie, percentage: float = 0.5):
        species.genomes.sort(key=lambda genome: genome.fitness)
        half= int(len(species.genomes)*percentage)
        species.genomes = species.genomes[half:]

    def remove_stale_species(self, max_staleness: int = 20):
        self.species = [species for species in self.species if species.staleness < max_staleness]


    def initialize_population(self,count:int,noInput:int,noOutput:int):
        input_nodes=[]
        for i in range(noInput):
            n=Node(idno=self.node_no,ntype='input')
            n.actfun= sigmoid_activation
            n.bias=random.randrange(-1,1)
            input_nodes.append(n)
            self.node_no +=1
        output_nodes=[]
        for i in range(noOutput):
            output_nodes.append(Node(idno=self.node_no,ntype='output'))
            self.node_no +=1
        
        self.genomes=[]
        for i in range(count):
            conn_list =[]
            for i in input_nodes:
                for j in  output_nodes:
                    conn_list.append(Connection(innovation_number=self.innovation_no,enable=True,input_id=i.id,out_id=j.id,weight=random.randrange(-1,1)))
                    self.innovation_no+=1
            self.genomes.append(Genome(node=input_nodes+output_nodes,connection=conn_list))

    def evolve(self, count: int, noInput: int, noOutput: int, noGeneration: int):
        self.initialize_population(count=count, noInput=noInput, noOutput=noOutput)

        for generation in range(noGeneration):
            print(f"Generation {generation + 1}")

            for genome in self.genomes:
                phenotype = Network(genome=genome)
                genome.fitness = evaluate_Fitness(phenotype)
                print(f"Genome Fitness: {genome.fitness}")

            self.species.clear()
            for genome in self.genomes:
                found_species = False
                for specie in self.species:
                    delta = specie.compatibilityDistance(genome)
                    if delta < compatibility_threshold:
                        specie.genomes.append(genome)
                        found_species = True
                        break
                if not found_species:
                    new_species = Specie(genome)
                    new_species.genomes.append(genome)
                    self.species.append(new_species)

            self.remove_stale_species()
            for s in self.species:
                self.remove_members(s)

            for s in self.species:
                for g in s.genomes:
                    g.fitness = g.fitness / len(s.genomes)

            total_adjacent_fitness = sum(g.fitness for s in self.species for g in s.genomes)
            print(f"Total Adjusted Fitness: {total_adjacent_fitness}")

            new_generation_genomes = []
            for s in self.species:
                species_adj_fitness = sum(g.fitness for g in s.genomes)
                num_offsprings = max(1, round(species_adj_fitness / total_adjacent_fitness * count))

                for _ in range(num_offsprings):
                    if len(s.genomes) >= 2:
                        parent1, parent2 = random.sample(s.genomes, 2)
                        if random.uniform(0, 1) > 0.5:
                            child_node = parent1.Crossover(parent2)
                        else:
                            child_node = random.choice([parent1, parent2])
                    else:
                        child_node = random.choice(s.genomes)

                    if child_node is not None:
                        if random.uniform(0, 1) > 0.5:
                            child_node.addNode()
                        if random.uniform(0, 1) > 0.5:
                            child_node.addConnection()
                        new_generation_genomes.append(child_node)


            self.genomes = new_generation_genomes
            self.current_generation = generation + 1

            
    

def evaluate_Fitness(phrnotype: Network):
    """
        this only made for the XOR 
    """
    input_list = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
    output_list = [0.0,1.0,1.0,0.0]
    total_error = 0.0
    correct= 0
    for i,out in zip(input_list,output_list):
        pred = phrnotype.activate(i)
        pred = pred[0]
        total_error+=(out-pred)**2
        if round(pred)==out:
            correct+=1
    return correct ** 2

            

    """
    todo: compleate this above create_new_species()
    """

if __name__ == "__main__":
    p = Population()
    p.evolve(count=20,noGeneration=50,noInput=2,noOutput=1)