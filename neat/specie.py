from neat.genome import Genome

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

    def compatibilityDistance(self, genome: Genome, c1=1.0, c2=1.0, c3=0.4):
        inGenome = {conn.innoNo: conn for conn in genome.conn}
        repGenome = {conn.innoNo: conn for conn in self.rep.conn}

        innovationNo1 = set(inGenome.keys())
        innovationNo2 = set(repGenome.keys())

        # Matching genes
        match_genes = innovationNo1 & innovationNo2

        # Disjoint genes
        disjoint_genes = (innovationNo1 - match_genes) | (innovationNo2 - match_genes)

        # Excess genes
        max_innovation1 = max(innovationNo1) if innovationNo1 else 0
        max_innovation2 = max(innovationNo2) if innovationNo2 else 0

        excess_genes = {
            inn for inn in innovationNo1 | innovationNo2
            if (inn > max_innovation2 and inn in innovationNo1) or
            (inn > max_innovation1 and inn in innovationNo2)
        }

        # Calculate the weight differences for matching genes
        if match_genes:
            W = sum(abs(inGenome[i].weight - repGenome[i].weight) for i in match_genes) / len(match_genes)
        else:
            W = 0.0

        # Normalization factor
        N = max(len(genome.conn), len(self.rep.conn))

        # Ensure N is not zero to avoid division by zero
        if N == 0:
            print(genome.conn,self.rep.conn)
            raise ValueError("Cannot calculate compatibility distance: both genomes have zero connections.")

        # Number of excess and disjoint genes
        E = len(excess_genes)
        D = len(disjoint_genes)

        # Compatibility distance
        delta = (c1 * E) / N + (c2 * D) / N + c3 * W
        return delta
