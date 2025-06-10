import random
from components import Node, Connection, Genome

COUNT = 20
input_list = [[0,0],[0,1],[1,0],[1,1]]
output_list = [0,1,1,0]



innovation_numbers = 0

node_numbers = 0

#In XOR there is 2 input and 1 output
start_nodes=[]
for n in range(2):
    start_nodes.append(Node(idno=node_numbers,ntype='input',actfun='',bias=random.random()))
    node_numbers+=1

out = [Node(idno=node_numbers,ntype='output',actfun='',bias=0)]
g=[]
for i in range(COUNT):
    genes_list =[]
    conn = Connection(innovation_number=innovation_numbers,enable=True,
                      input_id=start_nodes[random.randint(0,1)],out_id=output_list[0],
                      weight=random.random())
    innovation_numbers+=1
    genes_list.append(conn)
    g.append(Genome(connection=[genes_list],fitness=0,node=start_nodes+out))

