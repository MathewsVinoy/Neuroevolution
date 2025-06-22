from neat.activation import sigmoid_activation

class Node:
    def __init__(self,idno:int, ntype: str):
        self.id = idno
        self.type = ntype
        self.actfun = sigmoid_activation
        self.bias = 0