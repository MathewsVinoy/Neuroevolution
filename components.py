import random
from cross_over import Crossover
from activation import sigmoid_activation


innovation_numbers=[]
node_numbers=[]
innovation_no = 0

node_no = 0

compatibility_threshold = 3.0
POPULATIONA_SIZE =1

class Node:
    def __init__(self,idno:int, ntype: str, actfun, bias: float):
        self.id = idno
        self.type = ntype
        self.actfun =actfun
        self.bias = bias

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
    def __init__(self,node:list[Node], connection:list[Connection],fitness:float):
        self.node = node
        self.conn = connection
        self.fitness = fitness


    # mutation add connection
    def addConnection(self):
        if len(self.node)<2:
            raise RuntimeError("need 2 or more than 2 nodes")
        
        graph={}
        for c in self.conn:
            if c.enable == True:
                if c.inId not in graph:
                    graph[c.inId]=[]
                graph[c.inId].append(c.outId)
        for _ in range(len(self.nodes) * (len(self.nodes) - 1)):
            n1, n2 = random.sample(self.nodes, 2)
            if not self._creates_cycle(graph, n1.id, n2.id):    
                weight = random.randrange(-1,1)
                inn_no = len(innovation_numbers)
                innovation_numbers.append(inn_no)
                self.node.append(Connection(
                    input_id=n1,
                    out_id=n2,
                    enable=True,
                    innovation_number=inn_no,
                    weight=weight
                ))

    def _creates_cycle(self, graph, start, end):
        visited = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node == end:
                return True
            if node not in visited:
                visited.add(node)
                if node in graph:
                    for neighbor in graph[node]:
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
            actfun='',
            bias=0,
            idno=0,
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

    

class Network:
    """
    Network (Phenotype):
    Changes that can be seen / the changes that are visible.
    """
    def __init__(self, genome: Genome):
        self.node = genome.node
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
    def __init__(self, species: list[Specie]=[], genomes: list[Genome]=[]):
        self.species = species
        self.current_generation = 0
        self.genomes = genomes
    
    def remove_members(self, species: Specie, percentage: float = 0.5):
        species.genomes.sort(key=lambda genome: genome.fitness)
        half= int(len(species.genomes)*percentage)
        species.genomes = species.genomes[half:]

    def remove_stale_species(self, max_staleness: int = 20):
        self.species = [species for species in self.species if species.staleness < max_staleness]


    def initialize_population(self,count:int,noInput:int,noOutput:int):
        input_nodes=[]
        for i in range(noInput):
            input_nodes.append(Node(idno=node_numbers,ntype='input',actfun=sigmoid_activation))
            node_no +=1
        output_nodes=[]
        for i in range(noOutput):
            output_nodes.append(Node(idno=node_numbers,ntype='output'))
            node_no +=1
        
        self.genomes=[]
        for i in range(count):
            conn_list =[]
            for i in input_nodes:
                for j in  output_nodes:
                    conn_list.append(Connection(innovation_number=innovation_numbers,enable=True,input_id=i.id,out_id=j.id,weight=random.randrange(-1,1)))
                    innovation_no+=1
            self.genomes.append(Genome(node=input_nodes+output_nodes,connection=conn_list))

    def evolve(self,count:int,noInput:int,noOutput:int,noGeneration):

        self.initialize_population(count=count,noInput=noInput,noOutput=noOutput)

        for i in range(noGeneration):
            for genome in self.genomes:
                phenotype = Network(genome=genome)
                genome.fitness = evaluate_Fitness(phenotype)

            self.species.clear()
            for genome in self.genomes:
                found_species = False
                for specie in self.species:
                    delta = specie.compatibilityDistance(genome)
                    if delta < compatibility_threshold:
                        specie.genomes.append(genome)
                        found_species =True
                        break
                if not found_species:
                    new_species = Specie(genome)
                    new_species.genomes.append(genome)
            
            self.remove_stale_species()
            for s in self.species:
                self.remove_members(s)
            
            for s in self.species:
                for g in s.genomes:
                    g.fitness = g.fitness/len(s.genomes)
            
            new_generation_genomes = []
            total_adjacent_fitness = sum(g.fitness for s in self.species for g in s.genomes)

            for s in self.species:
                #  num_offspring_for_species = round(species.total_adjusted_fitness / total_adjusted_fitness * population_size)
                species_adj_fitness = sum(g.fitness for g in s.genomes)
                num_offsprings = round(species_adj_fitness/total_adjacent_fitness * count)
                for i in range(num_offsprings):
                    parent1, parent2 = random.sample(s.genomes, 2)
                    if random.randrange(-1,1) > 0.5:
                        child_node = Crossover(parent1=parent1,parent2=parent2)
                    else:
                        child_node = parent1
                    
                    if random.randrange(-1,1)> 0.5:
                        child_node.addNode()
                    if random.randrange(-1,1)> 0.5:
                        child_node.addConnection()
                    new_generation_genomes.append(child_node)
            self.genomes= new_generation_genomes
            self.current_generation = i+1
            
    

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