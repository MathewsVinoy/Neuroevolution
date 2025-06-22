from neat.genome import Genome

class Network:
    """
    Network (Phenotype):
    Changes that can be seen / the changes that are visible.
    """

    def __init__(self, genome: Genome):
        self.node = self._topological_sort(genome.nodes, genome.conn)
        self.conn = genome.conn

    def _topological_sort(self, nodes, connections):
        node_ids = {node.id for node in nodes}
        in_degree = {node.id: 0 for node in nodes}

        # Only count connections where outId is a valid node
        for conn in connections:
            if conn.enable and conn.outId in in_degree:
                in_degree[conn.outId] += 1
            elif conn.enable:
                print(f"Warning: Connection outId {conn.outId} not in node list.")

        queue = [node for node in nodes if in_degree[node.id] == 0]
        topo_order = []

        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for conn in connections:
                if conn.inId == node.id and conn.enable and conn.outId in in_degree:
                    in_degree[conn.outId] -= 1
                    if in_degree[conn.outId] == 0:
                        neighbor_node = next((n for n in nodes if n.id == conn.outId), None)
                        if neighbor_node is not None:
                            queue.append(neighbor_node)
                elif conn.enable and conn.outId not in in_degree:
                    print(f"Warning: Connection outId {conn.outId} not in node list.")

        if len(topo_order) != len(nodes):
            print("Nodes:", [n.id for n in nodes])
            print("Connections:", [(c.innoNo, c.inId, c.outId, c.enable) for c in connections])
            raise ValueError("The network contains a cycle or orphaned node and cannot be topologically sorted.")

        return topo_order

    def activate(self, input_vector):
        input_nodes = [node for node in self.node if node.type == 'i']
        output_nodes = [node for node in self.node if node.type == 'o']

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
            if node.type != 'i':  # Assuming 'i' stands for input nodes
                values[node.id] = node.actfun(total + node.bias)

        return [values[node.id] for node in output_nodes]

    def _build_network_graph(self, genome: Genome):
        pass
