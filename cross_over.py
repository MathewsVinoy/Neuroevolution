from genome import Genome
from components import Node, Connection
import random

def node_crossover(parent1:Node,parent2:Node)->Node:
    assert parent1.id==parent2.id
    bias = random.choice([parent1.bias,parent2.bias])
    act_fun = random.choice([parent1.actfun,parent2.actfun])
    return Node(idno=parent1.id,ntype=parent1.type,actfun=act_fun,bias=bias)
        

def connection_crossover(parent1: Connection,parent2:Connection)->Connection:
    assert parent1.innoNo == parent2.innoNo
    weight = random.choice([parent1.weight,parent2.weight])
    enable = random.choice([parent1.enable,parent2.enable])
    return Connection(innovation_number=parent1.innoNo,input_id=parent1.inId,out_id=parent1.outId,weight=weight,enable=enable)



def Crossover(parent1: Genome, parent2: Genome) -> Genome:
    conn_dict1 = {conn.innoNo: conn for conn in parent1.conn}
    conn_dict2 = {conn.innoNo: conn for conn in parent2.conn}

    all_conn = sorted(set(conn_dict1.keys()) | set(conn_dict2.keys()))


    if parent1.fitness > parent2.fitness:
        fitness = parent1.fitness
    else:
        fitness = parent2.fitness


    new_conn = []
    for i in all_conn:
        conn1 = conn_dict1.get(i)
        conn2 = conn_dict2.get(i)
        print(conn1)

        if conn1 and conn2:
            new_conn.append(connection_crossover(conn1, conn2))
        elif conn1:
            new_conn.append(conn1)
        elif conn2:
            new_conn.append(conn2)

    return Genome(connection=new_conn,node=[],fitness=fitness)

    


    





nodes1 = [Node(idno=3,ntype='0',actfun='tan',bias=8),Node(idno=1,ntype='i',actfun='sin',bias=10),Node(idno=4,ntype='i',actfun='sin',bias=10),Node(idno=2,ntype='h',actfun='cos',bias=9)]
nodes2 = [Node(idno=1,ntype='i',actfun='sin',bias=4),Node(idno=2,ntype='h',actfun='cos',bias=8),Node(idno=3,ntype='0',actfun='tan',bias=7)]

connn1 = [Connection(enable=True,input_id=1,out_id=2,innovation_number=1,weight=10),Connection(enable=True,input_id=2,out_id=3,innovation_number=3,weight=10),Connection(enable=True,input_id=2,out_id=4,innovation_number=4,weight=10),]
connn2 = [Connection(enable=True,input_id=1,out_id=2,innovation_number=1,weight=10),Connection(enable=True,input_id=2,out_id=3,innovation_number=3,weight=10)]
if __name__ == "__main__":
    Crossover(
        parent1=Genome(node=nodes1,fitness=10,connection=connn1),
        parent2=Genome(node=nodes2,fitness=2,connection=connn2)
    )