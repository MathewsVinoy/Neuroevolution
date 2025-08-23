import random
from neat.fitness import save_model
from neat.config import Config
from neat.node import Node
from neat.connection import Connection
from neat.genome import Genome
from neat.specie import Specie
from neat.network import Network
from neat.activation import exp_activation

class Population(object):
    evaluate_Fitness = None

    
    def __init__(self):
        self.species = []
        self.current_generation = -1
        self.genomes = []
        self.best_fitness = []
        self.avg_fitness =[]
        self.species_log = []
        self.create_population()
        
    
    def create_population(self):
        self.genomes =[]
        for i in range(Config.pop_size):
            g = Genome.create_minimally_connected()
            self.genomes.append(g)


    def speciate(self):
        # print(self.genomes)
        for genome in self.genomes:
            found = False
            # print(genome)
            for s in self.species:
                if genome.distance(s.rep) < Config.compatibility_threshold:
                    s.add(genome)
                    found = True
                    break
            if not found:
                self.species.append(Specie(rep=genome))

        for s in self.species:
            if len(s) == 0:
                self.species.remove(s)

        self.set_compatibility_threshlod()

    def set_compatibility_threshlod(self):
        if len(self.species) > Config.species_size:
            Config.compatibility_threshold += Config.compatibility_change
        elif len(self.species) < Config.species_size:
            if Config.compatibility_threshold > Config.compatibility_change:
                Config.compatibility_threshold -= Config.compatibility_change
    def avg_fitness_fn(self):
        sum = 0.0
        for g in self.genomes:
            sum += g.fitness
        
        
        return sum/len(self.genomes)
    
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
            total_average += s

        for i, s in enumerate(self.species):
            s.spawn_amount = int(round((species_stats[i]*Config.pop_size/total_average)))
        
    def log_species(self):
        higher = max([s.id for s in self.species])
        temp=[]
        for i in range(higher):
            found_species = False
            for s in self.species:
                if i == s.id:
                    temp.append(len(s.genomes))     
                    found_species = True
                    break
            if not found_species:
                temp.append(0)
        self.species_log.append(temp) 


    def evolve(self, no):
        for _ in range(no):
            self.current_generation += 1
            print('\n ****** Running generation ', self.current_generation, ' ****** \n')
            self.evaluate_Fitness()
            self.speciate()
            self.best_fitness.append(max(self.genomes, key=lambda g: g.fitness))
            self.avg_fitness.append(self.avg_fitness_fn())
            best = self.best_fitness[-1]

            for s in self.species:
                s.hasBest = False
                if best.species_id == s.id:
                    s.hasBest = True
            save_model(best)

            # Remove the worst performing genomes
            for s in self.species[:]:
                if s.no_improvement_age > Config.max_stagnation:
                    if not s.hasBest:
                        self.species.remove(s)
                        for g in self.genomes[:]:
                            if g.species_id == s.id:
                                self.genomes.remove(g)
                if s.no_improvement_age > 2 * Config.max_stagnation:
                    self.species.remove(s)
                    for g in self.genomes[:]:
                        if g.species_id == s.id:
                            self.genomes.remove(g)

            self.compute_spawn_levels()
            for s in self.species:
                if s.spawn_amount == 0:
                    for g in self.genomes[:]:
                        if g.species_id == s.id:
                            self.genomes.remove(g)
            

            self.log_species()
            new_population = []
            for s in self.species:
                new_population = s.reproduce()
            

            

            fill = Config.pop_size - len(new_population)
            print(f"Population fill required: {fill}")  # Debugging statement

            if fill < 0:
                new_population = new_population[:Config.pop_size]
            if fill > 0:
                while fill > 0:
                    parent1 = random.choice(self.genomes)
                    found = False
                    for c in self.genomes:
                        if c.species_id == parent1.species_id:
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
            assert Config.pop_size == len(new_population), 'Different population sizes!'
            self.species =[]
            self.genomes = new_population
