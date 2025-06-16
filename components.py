import random
from activation import sigmoid_activation


innovation_numbers=[]
node_numbers=[]
innovation_no = 0

node_no =0

compatibility_threshold = 3.0
POPULATIONA_SIZE =1




def evaluate_Fitness(phrnotype: Network):
    """
        this only made for the XOR 
    """
    input_list = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
    output_list = [0.0,1.0,1.0,0.0]
    total_error = 0.0
    correct= 0
    for i,out in zip(input_list,output_list):
        pred = phrnotype.activate(i)
        pred = pred[0]
        total_error+=(out-pred)**2
        if round(pred)==out:
            correct+=1
    return correct ** 2

            

    """
    todo: compleate this above create_new_species()
    """

if __name__ == "__main__":
    p = Population()
    p.evolve(count=20,noGeneration=50,noInput=2,noOutput=1)