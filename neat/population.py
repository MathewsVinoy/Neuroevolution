import random
from neat.fitness import save_model
from neat.config import Config
from neat.node import Node
from neat.connection import Connection
from neat.genome import Genome
from neat.specie import Specie
from neat.network import Network
from neat.activation import sigmoid_activation

class Population(object):
    evaluate_Fitness = None

    
    def __init__(self):
        self.species = []
        self.current_generation = -1
        self.genomes = []
        self.best_fitness = []
        self.avg_fitness =[]
        self.create_population()
        
    
    def create_population(self):
        self.genome =[]
        for i in range(Config.pop_size):
            g = Genome.create_minimally_connected()
            self.genome.append(g)

            
    def remove_members(self, species: Specie, percentage: float = 0.5):
        species.genomes.sort(key=lambda genome: genome.fitness)
        half= int(len(species.genomes)*percentage)
        species.genomes = species.genomes[half:]

    def remove_stale_species(self, max_staleness: int = 20):
        self.species = [species for species in self.species if species.staleness < max_staleness]


    def initialize_population(self,count:int,noInput:int,noOutput:int):
        
        input_nodes=[]
        for i in range(noInput):
           
            input_nodes.append(Node(idno=Config.node_no,ntype='i'))
            Config.node_no +=1
        output_nodes=[]
        for i in range(noOutput):
            n=Node(idno=Config.node_no,ntype='o')
            n.bias = random.random()
            output_nodes.append(n)

            Config.node_no +=1
        
        self.genomes=[]
        for _ in range(count):
            conn_list =[]
            for i in output_nodes:
                for j in  input_nodes:
                    inno=Config.innnovationTracker(j.id,i.id)
                    conn_list.append(Connection(innovation_number=inno,enable=True,input_id=j.id,out_id=i.id,weight=random.random()))
            self.genomes.append(Genome(node=input_nodes+output_nodes,connection=conn_list))

    def speciate(self):
        for genome in self.genomes:
            found = False
            for s in self.species:
                if genome.distance(s.rep) < Config.compatibility_threshold:
                    s.add(genome)
                    found = True
            if not found:
                self.species.append(Specie(rep=genome))

        self.set_compatibility_threshlod()

    def set_compatibility_threshlod(self):
        if len(self.species) > Config.species_size:
            Config.compatibility_threshold += Config.compatibility_change
        elif len(self.__species) < Config.species_size:
            if Config.compatibility_threshold > Config.compatibility_change:
                Config.compatibility_threshold -= Config.compatibility_change
    def avg_fitness_fn(self):
        sum = 0.0
        for g in self.genome:
            sum += g.fitness
        
        return sum/len(self.genome)
    
    def compute_spawn_levels(self):
        species_stats =[]
        for s in self.species:
            if s.age < Config.youth_threshold:
                species_stats.append(s.average_fitness()*Config.youth_boost)
            elif s.age > Config.old_threshold:
                species_stats.append(s.average_fitness()*Config.old_penalty)
            else:
                species_stats.append(s.average_fitness())
        total_average = 0.0 
        for s in species_stats:
            total_average +=5

        for i, s in enumerate(self.species):
            s.spawn_amount = int(round((species_stats[i]*Config.pop_size/total_average)))
        
    def log_species(self):
        pass

    def evolve(self, no):
        for _ in range(no):
            self.current_generation +=1
            self.evaluate_Fitness()
            self.speciate()
            self.best_fitness.append(max(self.genomes))
            self.avg_fitness.append(self.avg_fitness_fn())
            best = self.best_fitness[-1]
            for s in self.species:
                s.hasBest = False
                if best.id in s.genomes:
                    s.hasBest = True
            
            save_model(best)

            # remove the based performing genomes
            for s in self.species[:]:
                if s.no_improvement_age > Config.max_stagnation:
                    if not s.hasBest:
                        self.species.remove(s)
                        for g in self.genome[:]:
                            if g in s.genomes:
                                self.remove(g)
            
            for s in self.species:
                if s.no_inmprovement_age > 2 *Config.max_stagnation:
                    self.species.remove(s)

                    for g in self.genomes:
                        if g in s.genomes:
                            self.genomes.remove(g)

            self.compute_spawn_levels()
            for s in self.species:
                if s.spawn_amount == 0:
                    for g in self.genomes:
                        if g in s.genomes:
                            self.genomes.remove(s)

            self.log_species()