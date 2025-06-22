from neat.network import Network

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