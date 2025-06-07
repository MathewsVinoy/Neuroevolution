import random


innovation_numbers=[]
node_numbers=[]

class Node:
    def __init__(self,idno:int, ntype: str, actfun, bias: float):
        self.id = idno
        self.type = ntype
        self.actfun =actfun
        self.bias = bias
        self.value = 0.0

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
    def __init__(self,node:list[Node], connection:list[Connection],fitness:float):
        self.node = node
        self.conn = connection
        self.fitness = fitness


    # mutation add connection
    def addConnection(self,in_node,out_node):
        weight = random.randrange(-1,1)
        inn_no = len(innovation_numbers)
        innovation_numbers.append(inn_no)
        conn = Connection(
            input_id=in_node,
            out_id=out_node,
            enable=True,
            innovation_number=inn_no,
            weight=weight
        )
    
    # mutation add Node
    def addNode(self, old_conn: Connection,type):
        old_conn.enable=False
        no = len(node_numbers)
        node_numbers.append(no)
        new_node = Node(
            ntype=type,
            actfun='',
            bias=0,
            idno=0,
        )
        inn_no = len(innovation_numbers)
        innovation_numbers.append(inn_no)
        conn1 = Connection(
            input_id=old_conn.inId,
            enable=True,
            innovation_number=inn_no,
            out_id= new_node.id,
            weight=1,
        )
        inn_no = len(innovation_numbers)
        innovation_numbers.append(inn_no)
        conn2 = Connection(
            input_id=new_node.id,
            enable=True,
            innovation_number=inn_no,
            out_id= old_conn.outId,
            weight=old_conn.weight,
        )

    

class Network:
    """
        Network (Phenotype):
         changes that can been seen / the changes that are visible
    """
    def __init__(self,node, connection):
        self.node = node
        self.conn = connection



class Specie:
    """
    species is represented by a random genome inside the species from the previous generation
    """

    def __init__(self,id,genomes: list[Genome],rep: Genome):
        self.genomes= genomes
        self.id = id
        self.rep = rep

    def compatibilityDistance(self, genome: Genome,c1=1.0, c2=1.0, c3=0.4):
        inGenome = {conn.innoNo: conn for conn in genome.conn}
        repGenome = {conn.innoNo: conn for conn in self.rep.conn}

        innovationNo1 = set(inGenome.keys())
        innovationNo2 = set(repGenome.keys())

        match_genes =  innovationNo1 & innovationNo2
        disjoint_genes = (innovationNo1^innovationNo2)-(innovationNo1|innovationNo2).difference(match_genes)
        exec_genes = {
            inn for inn in innovationNo1 | innovationNo2 
            if(inn> max(innovationNo1)and inn in innovationNo2)or 
            (inn >max(innovationNo2)and inn in innovationNo1)
        }
        if match_genes:
            W= sum(abs(inGenome[i].weight-repGenome[i].weight)for i in match_genes)
        else:
            W=0.0

        N= max(len(genome.conn),len(self.rep.conn))
        E = len(exec_genes)
        D = len(disjoint_genes)

        delta = (c1*E)/N + (c2*D)/N +(c3* W)
        return delta

class Population:
    def __init__(self, species: list[Specie]):
        self.species = species
        self.id =0

"""
todo:fitness shereing with its equation
"""
