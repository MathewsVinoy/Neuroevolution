innovation_numbers = []
node_numbers = []

class Node:
    def __init__(self,idno, ntype, actfun, bias):
        self.id = idno
        self.type = ntype
        self.actfun =actfun
        self.bias = bias

class Connection:
    """
    Connection:
     This class is used to creat an istance of the connection between 2 nodes.

     self.enable => (dtype = boolen) diable when new node is added then added to the last of the Genome
     innovation numbers =>(dtype = int) unique number reprsinting each connection
    """
    
    def __init__(self,input_id,out_id,weight,enable,innovation_number):
        self.inId = input_id
        self.outId = out_id
        self.weight = weight
        self.enable = enable  
        self.innoNo = innovation_number 


class Genome:
    """
        Genotype(Genome):
            things the are not visible
    """
    def __init__(self,node:list, connection:list,fitness:float):
        self.node = node
        self.conn = connection
        self.fitness = fitness


class Network:
    """
        Network (Phenotype):
         changes that can been seen / the changes that are visible
    """
    def __init__(self,node, connection):
        self.node = node
        self.conn = connection