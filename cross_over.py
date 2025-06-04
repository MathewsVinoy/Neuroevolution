import random
from genome import Genome

def CrossOver(parent1:Genome,parent2:Genome):
    nodes = parent1.node  + parent2.node
    node = tuple(*nodes,)
    conns =  parent1.conn  + parent2.conn
    conn = tuple(*conns,)
    new_genome = Genome(node=node,connection=conn,fitness=random.randrange(-1,1))