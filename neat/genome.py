from random import choice, sample, uniform
from typing import List

# Assuming Node and Connection classes are defined in neat.node and neat.connection respectively
from neat.node import Node
from neat.connection import Connection
from neat.config import Config
from neat.graph import cycle_check

class Genome:
    """
    Genotype (Genome):
        Represents the genetic information of a neural network.
    """

    def __init__(self, node, connection):
        self.nodes = node
        self.conn = connection
        self.fitness = 0

    def addConnection(self):
        """
        Adds a new connection between two nodes if it doesn't create a cycle or violate other constraints.
        """
        c = Config()

        if len(self.nodes) < 2:
            return  # Not enough nodes to form a connection

        n1 = choice(self.nodes)
        n2 = choice(self.nodes)

        # Self-loop check
        if n1.id == n2.id:
            return

        # Enforce direction based on node type
        if n1.type == 'output' and n2.type != 'output':
            n1, n2 = n2, n1
        elif n2.type == 'input' and n1.type != 'input':
            n1, n2 = n2, n1
        elif n1.type == 'output' and n2.type == 'input':
            return
        elif n1.type == 'output' and n2.type == 'output':
            return
        elif n1.type == 'input' and n2.type == 'input':
            return

        # Check if connection already exists
        exists = any(
            conn.inId == n1.id and conn.outId == n2.id
            for conn in self.conn if conn.enable
        )
        if exists:
            return

        # Check for cycle (if feedforward only)
        if cycle_check(self.conn, (n1.id, n2.id)):
            return

        # All checks passed — add the connection
        weight = uniform(-1, 1)
        new_connection = Connection(
            input_id=n1.id,
            out_id=n2.id,
            enable=True,
            innovation_number=c.innovation_no,
            weight=weight
        )
        c.innovation_no += 1
        self.conn.append(new_connection)

    def _creates_cycle(self, graph, start, end):
        """
        Helper method to check if adding a connection creates a cycle.
        """
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

    def removeConnection(self):
        """
        Removes a random connection from the genome.
        """
        if not self.conn:
            return
        conn_to_remove = choice(self.conn)
        if not conn_to_remove.enable:
            return
        conn_to_remove.enable = False
        self.conn.remove(conn_to_remove)
        del conn_to_remove

    def removeNode(self):
        """
        Removes a random node from the genome and disables all its connections.
        """
        if not self.nodes:
            return
        node_to_remove = choice(self.nodes)
        if node_to_remove.type in ('output', 'input'):
            return
        conn = [c for c in self.conn if c.inId == node_to_remove.id or c.outId == node_to_remove.id]
        for c in conn:
            c.enable = False
            self.conn.remove(c)
            del c
        self.nodes.remove(node_to_remove)
        del node_to_remove

    def addNode(self):
        """
        Adds a new node to the genome by splitting an existing connection.
        """
        c = Config()
        if not self.conn:
            return
        old_conn = choice(self.conn)
        old_conn.enable = False
        new_node = Node(
            ntype='hidden',  # Assuming 'hidden' is the type for new nodes
            idno=c.node_no,
        )
        c.node_no += 1
        self.nodes.append(new_node)
        conn1 = Connection(
            input_id=old_conn.inId,
            out_id=new_node.id,
            enable=True,
            innovation_number=c.innovation_no,
            weight=1,
        )
        c.innovation_no += 1
        self.conn.append(conn1)
        conn2 = Connection(
            input_id=new_node.id,
            out_id=old_conn.outId,
            enable=True,
            innovation_number=c.innovation_no,
            weight=old_conn.weight,
        )
        c.innovation_no += 1
        self.conn.append(conn2)

    def node_crossover(self, parent1: Node, parent2: Node) -> Node:
        """
        Performs crossover between two nodes to produce a new node.
        """
        assert parent1.id == parent2.id
        bias = choice([parent1.bias, parent2.bias])
        act_fun = choice([parent1.actfun, parent2.actfun])
        n = Node(idno=parent1.id, ntype=parent1.type)
        n.actfun = act_fun
        n.bias = bias
        return n

    def connection_crossover(self, parent1: Connection, parent2: Connection) -> Connection:
        """
        Performs crossover between two connections to produce a new connection.
        """
        assert parent1.innoNo == parent2.innoNo
        weight = choice([parent1.weight, parent2.weight])
        enable = choice([parent1.enable, parent2.enable])
        return Connection(
            innovation_number=parent1.innoNo,
            input_id=parent1.inId,
            out_id=parent1.outId,
            weight=weight,
            enable=enable
        )

    def Crossover(self, parent2):
        """
        Performs crossover between two genomes to produce a new genome.
        """
        conn_dict1 = {conn.innoNo: conn for conn in self.conn}
        conn_dict2 = {conn.innoNo: conn for conn in parent2.conn}

        all_conn = sorted(set(conn_dict1.keys()) | set(conn_dict2.keys()))

        new_conn = []
        for i in all_conn:
            conn1 = conn_dict1.get(i)
            conn2 = conn_dict2.get(i)
            if conn1 and conn2:
                new_conn.append(self.connection_crossover(conn1, conn2))
            elif conn1:
                new_conn.append(conn1)
            elif conn2:
                new_conn.append(conn2)

        new_nodes = []
        for n1 in self.nodes:
            for n2 in parent2.nodes:
                if n1.id == n2.id:
                    new_nodes.append(self.node_crossover(n1, n2))

        child_genome = Genome(node=new_nodes, connection=new_conn)
        return child_genome
