from random import choice, randrange
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
        self.next_node_id = Config.node_no

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
        assert parent1.inId == parent2.inId
        assert parent1.outId == parent2.outId
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
        conn1 = {conn.innoNo: conn for conn in self.conn}
        conn2 = {conn.innoNo: conn for conn in parent2.conn}
        new_conn =[]
        for key , v in conn1.items():
            conn = conn2.get(key)
            if conn is None:
                new_conn.append(v)
            else:
                new_conn.append(self.connection_crossover(v,conn))
        
        node1 = {node.id: node for node in self.nodes}
        node2 = {node.id: node for node in parent2.nodes}
        new_nodes = []
        for key, v in node1.items():
            node  = node2.get(key)
            if node is None:
                new_nodes.append(v)
            else:
                new_nodes.append(self.node_crossover(v, node))

        self.next_node_id = max(self.next_node_id,parent2.next_node_id)
        self.nodes=new_nodes
        self.conn=new_conn

            
        
        
    

    
    def addConnection(self):
        if len(self.nodes) < 2:
            return
        n1 = choice(self.nodes)
        n2 = choice(self.nodes)

        if n1.id == n2.id:
            return
        
        if n1.type == 'o' and n2.type == 'o':
            return
        if n1.type == 'i' and n2.type == 'i':
            return
        if n1.type == 'o' or n2.type =='i':
            n1,n2 =n2, n1
        
        for c in self.conn:
            if (c.inId == n1.id and c.outId == n2.id and c.enable) or (c.inId == n2.id and c.outId == n1.id and c.enable):
                return
            
        if cycle_check(self.conn, (n1.id, n2.id)):
            return
        
        inno = Config.innnovationTracker(n1.id,n2.id)
        new_connection = Connection(
            input_id=n1.id,
            out_id=n2.id,
            enable=True,
            innovation_number=inno,
            weight=randrange(-1, 1)
        )
        self.conn.append(new_connection)

    def removeConnection(self):
        if len(self.conn) < 2:
            return
        conn_to_remove = choice(self.conn)
        conn_to_remove.enable = False
        self.conn.remove(conn_to_remove)
        del conn_to_remove

    def addNode(self):
        if not self.conn:
            return
        old_conn = choice(self.conn)
        old_conn.enable = False
        new_node = Node(idno=self.next_node_id, ntype='h')
        self.next_node_id += 1
        self.nodes.append(new_node)
        new_conn1 = Connection(
            input_id=old_conn.inId,
            out_id=new_node.id,
            enable=True,
            innovation_number=Config.innnovationTracker(old_conn.inId, new_node.id),
            weight=old_conn.weight
        )
        new_conn2 = Connection(
            input_id=new_node.id,
            out_id=old_conn.outId,
            enable=True,
            innovation_number=Config.innnovationTracker(new_node.id, old_conn.outId),
            weight=1
        )
        self.conn.append(new_conn1)
        self.conn.append(new_conn2)
        self.conn.remove(old_conn)
        del old_conn

    def removeNode(self):
        if not self.nodes:
            return
        node_to_remove = choice(self.nodes)
        if node_to_remove.type == 'o' or node_to_remove.type == 'i':
            return
        conn = [ c for c in self.conn if c.inId == node_to_remove.id or c.outId == node_to_remove.id]
        if len(conn) >= len(self.conn):
            return
        for c in conn:
            c.enable = False
            self.conn.remove(c)
            del c
        self.nodes.remove(node_to_remove)
        del node_to_remove
        

