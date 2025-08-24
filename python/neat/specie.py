from neat.genome import Genome
from neat.config import Config
from random import choice

class Specie:
    """
    species is represented by a random genome inside the species from the previous generation
    """
    id = 0

    def __init__(self,rep: Genome):
        self.genomes= []
        self.id = self.get_new_id()
        self.age = 0
        self.rep = rep
        self.staleness = 0.0
        self.hasBest = False
        self.no_improvement_age = 0
        self.add(rep)
        self.compatibility_threshold = 3.0
        self.last_avg_fitness = 0
        self.spawn_amount = 0

    def __str__(self):
        return f"rep--> {self.rep} \n genomes --> {self.genomes}"

    def add(self,genome: Genome):
        genome.species_id = self.id
        self.genomes.append(genome)
        self.rep = choice(self.genomes)
    
    def __len__(self):
        return len(self.genomes)

    @classmethod
    def get_new_id(cls, previous_id=None):
        if previous_id is None:
            cls.id += 1
            return cls.id
        else:
            return previous_id


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

        
    def reproduce(self):
        offspring = []
        self.age += 1

        # assert self.spawn_amount > 0 

        self.genomes.sort(key=lambda g: g.fitness, reverse=True)

        if Config.elitism:
            offspring.append(self.genomes[0])
            self.spawn_amount -= 1
        



        survivors = int(round(len(self)*Config.survival_threshold))
        if survivors > 0:
            self.genomes = self.genomes[:survivors]
        else:
            self.genomes = self.genomes[:1]
        
        while(self.spawn_amount> 0):
            self.spawn_amount -=1
            if len(self) > 1:
                parent1 =  choice(self.genomes)
                parent2 = choice(self.genomes)
                assert parent1.species_id == parent2.species_id
                parent1.crossover(parent2)
                parent1.mutate()
                # print("1")
                if parent1 == None:
                    print(parent1)
                offspring.append(parent1)
            else:
                parent1 = self.genomes[0]
                parent1.crossover(parent1)
                parent1.mutate()
                # print("2")
                if parent1 == None:
                    print(parent1)
                offspring.append(parent1)
    
        # print("offspring=",offspring)
        
        self.genomes = []
        self.rep = choice(offspring)

        return offspring