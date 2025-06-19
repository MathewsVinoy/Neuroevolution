import random
from neat.fitness import evaluate_Fitness
from neat.config import Config
from neat.node import Node
from neat.connection import Connection
from neat.genome import Genome
from neat.specie import Specie
from neat.network import Network
from neat.activation import sigmoid_activation

class Population:
    def __init__(self):
        self.species = []
        self.current_generation = 0
        self.genomes = []
            
    def remove_members(self, species: Specie, percentage: float = 0.5):
        species.genomes.sort(key=lambda genome: genome.fitness)
        half= int(len(species.genomes)*percentage)
        species.genomes = species.genomes[half:]

    def remove_stale_species(self, max_staleness: int = 20):
        self.species = [species for species in self.species if species.staleness < max_staleness]


    def initialize_population(self,count:int,noInput:int,noOutput:int):
        
        input_nodes=[]
        for i in range(noInput):
            n=Node(idno=Config.node_no,ntype='input')
            n.actfun= sigmoid_activation
            n.bias=random.randrange(-1,1)
            input_nodes.append(n)
            Config.node_no +=1
        output_nodes=[]
        for i in range(noOutput):
            output_nodes.append(Node(idno=Config.node_no,ntype='output'))
            Config.node_no +=1
        
        self.genomes=[]
        for i in range(count):
            Config.innovation_no = 0
            conn_list =[]
            for i in output_nodes:
                for j in  input_nodes:
                    conn_list.append(Connection(innovation_number=Config.innovation_no,enable=True,input_id=j.id,out_id=i.id,weight=random.randrange(-1,1)))
                    Config.innovation_no+=1
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
                    if delta < Config().compatibility_threshold:
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
                        if random.random() > 0.5:
                            child_node = parent1.Crossover(parent2)
                        else:
                            child_node = random.choice([parent1, parent2])
                    else:
                        child_node = random.choice(s.genomes)

                    if child_node is not None:
                        if random.random() < 0.5:
                            child_node.removeConnection()
                        if random.random() < 0.5:
                            child_node.addConnection()
                        if random.random() <0.2:
                            child_node.removeNode()
                        if random.random() < 0.2:
                            child_node.addNode()
                        new_generation_genomes.append(child_node)


            self.genomes = new_generation_genomes
            self.current_generation = generation + 1

            
    