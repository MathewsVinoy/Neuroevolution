from neat.genome import Genome


class Network:
    """
    Network (Phenotype):
    Changes that can be seen / the changes that are visible.
    """
    def __init__(self,genome: Genome):
        self.nodes = genome.nodes
        self.conns = genome.conn

    def _build_network_graph(self, genome:Genome):
        pass