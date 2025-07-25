from neat.genome import Genome
from light.linear import Linear
from neat.config import Config

class Network:
    """
    Network (Phenotype):
    Represents the expressed neural network constructed from a genome.
    """

    def __init__(self, genome: Genome):
        self.nodes = genome.nodes
        self.conn = [c for c in genome.conn if c.enable]  # Only enabled connections

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
                if c.outId == node.id:
                    input_node = next((n for n in self.nodes if n.id == c.inId), None)
                    if input_node is not None:
                        total_input += c.weight * input_node.output

            node.output = node.actfun(total_input)

        # Collect outputs
        output_values = [n.output for n in self.nodes if n.type == 'OUTPUT']
        return output_values
