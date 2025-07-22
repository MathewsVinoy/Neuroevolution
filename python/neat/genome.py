from random import choice, randrange, gauss, random, randint
from neat.node import Node
from neat.connection import Connection
from neat.config import Config
from neat.graph import cycle_check
import math

class Genome(object):
    """
    Genotype (Genome):
        Represents the genetic information of a neural network.
    """

    def __init__(self, node, connection):
        self.nodes = node
        self.conn = connection
        self.fitness = 0
        self.species_id = None
        self.next_node_id = Config.input_nodes + Config.hidden_nodes + Config.output_nodes

    def __str__(self):
        return f"conn-> {self.conn} \n Nodes -> {self.nodes} "

    def mutate(self):
        if random() < Config.prob_addnode:
            self.addNode()
        elif random() < Config.prob_addconn:
            self.addConnection()
        else:
            for c in self.conn:
                c.mutate()
            for n in self.nodes:
                if n.type != 'INPUT':
                    n.mutate()

    def distance(self, other):
        if len(self.conn) > len(other.conn):
            parent1 = self.conn
            parent2 = other.conn
        else:
            parent1 = other.conn
            parent2 = self.conn

        weight_diff =0
        matching = 0
        disjoint = 0
        excess = 0

        max_cg_parent2 = max(c.innoNo for c in parent2)
        cg2 = {c.innoNo:c for c in parent2}
        for c1 in parent1:
            try:
                c2 = cg2[c1.innoNo]
            except KeyError:
                if c1.innoNo > max_cg_parent2:
                    excess += 1
                else:
                    disjoint +=1
            else:
                weight_diff += math.fabs(c1.weight- c2.weight)

        disjoint += len(parent2) - matching

        distance = Config.excess_coeficient * excess + Config.disjoint_coeficient * disjoint

        if matching > 0:
            distance += Config.weight_coeficient * (weight_diff/matching)

        return distance
    

    @staticmethod
    def create_minimally_connected():
        input_nodes = []
        id =0
        for i in range(Config.input_nodes):
            input_nodes.append(Node(id,'INPUT'))
            id += 1

        output_nodes = []
        for i in range(Config.output_nodes):
            output_nodes.append(Node(id,'OUTPUT'))
            id += 1
        
        nodes=input_nodes+output_nodes
        conn=[]
        assert len(nodes) == id
        for node in nodes:
            if node.type != 'OUTPUT':
                continue
            input_node = choice(input_nodes)
            wight = gauss(0,Config.weight_stdev)
            conn.append(Connection(input_id=input_node.id,out_id=node.id,weight=wight,enable=True))
        # print(conn)
        
        return Genome(node=nodes, connection=conn)


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
        out = Connection(
            input_id=parent1.inId,
            out_id=parent1.outId,
            weight=weight,
            enable=enable
        )
        out.innoNo = parent1.innoNo
        return out

    def crossover(self, parent2):
        """
        Performs crossover between two genomes to produce a new genome.
        """
        assert self.species_id == parent2.species_id

        conn1 = {conn.innoNo: conn for conn in self.conn}
        conn2 = {conn.innoNo: conn for conn in parent2.conn}
        new_conn =[]
        for key , v in conn1.items():
            conn = conn2.get(key)
            if conn != None:
                if key == conn.innoNo:
                    new_conn.append(self.connection_crossover(v,conn))
                else:
                    new_conn.append(v.copy())
            else:
                new_conn.append(v.copy())
        self.conn=new_conn
        
        node1 = {node.id: node for node in self.nodes}
        node2 = {node.id: node for node in parent2.nodes}
        new_nodes = []
        # print(node2)
        for key, v in node1.items():
            # print(key)
            try:
               node = node2.get(key)
            except KeyError:
                new_nodes.append(v)
            else:
                if node != None:
                    if key ==node.id:
                        new_nodes.append(self.node_crossover(v, node))
                    else:
                        new_nodes.append(v)
                else:
                    new_nodes.append(v)
                
        # print(new_nodes)
        self.next_node_id = max(self.next_node_id,parent2.next_node_id)
        self.nodes=new_nodes
        


    def addConnection(self):
        num_h = len([h for h in self.nodes if h.type == 'HIDDEN'])
        num_o = len(self.nodes) - Config.input_nodes - num_h

        total_possible_conns = (num_h+num_o)*(Config.input_nodes+num_h) -sum(range(num_h+1)) 
        rem_conns = total_possible_conns - len(self.conn)
        conns = [(c.inId,c.outId) for c in self.conn]
        if rem_conns >0:
            n = randint(0,rem_conns-1)
            count =0
            for in_node in (self.nodes[:Config.input_nodes]+self.nodes[-num_h:]):
                for out_node in self.nodes[Config.input_nodes:]:
                    if (in_node,out_node) not in conns and self.__is_connection_feedforward(in_node, out_node):
                        if count == n:
                            weight = gauss(0,1)
                            cg = Connection(in_node,out_node,weight,enable=True)
                            self.conn.append(cg)
                            return
                        else:
                            count += 1
    
    def __is_connection_feedforward(self, in_node, out_node):
        
        return in_node.type == 'INPUT' or out_node.type == 'OUTPUT' or \
          in_node.id <out_node.id



    def removeConnection(self):
        if len(self.conn) < 2:
            return
        conn_to_remove = choice(self.conn)
        conn_to_remove.enable = False
        self.conn.remove(conn_to_remove)
        del conn_to_remove

    def addNode(self):
        old_conn = choice(self.conn)
        new_node = Node(idno=len(self.nodes), ntype='HIDDEN')
        self.nodes.append(new_node)
        old_conn.enable = False
        new_conn1 = Connection(
            input_id=old_conn.inId,
            out_id=new_node.id,
            enable=True,
            weight=1
        )
        new_conn2 = Connection(
            input_id=new_node.id,
            out_id=old_conn.outId,
            enable=True,
            weight=old_conn.weight
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
        

