from genome import Genome

innovation_numbers = []
node_numbers = []

class Node:
    def __init__(self,idno:int, ntype: str, actfun, bias: float):
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


class Network:
    """
        Network (Phenotype):
         changes that can been seen / the changes that are visible
    """
    def __init__(self,node, connection):
        self.node = node
        self.conn = connection



class Species:
    def __init__(self,id,genomes: list[Genome]):
        self.genomes= genomes
        self.id = id  