import random
from components import Node, Connection, innovation_numbers, node_numbers

class Mutation:
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

   