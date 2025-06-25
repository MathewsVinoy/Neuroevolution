from neat.network import Network
import pickle
from pathlib import Path
from random import randrange

def evaluate_Fitness(phrnotype: Network):
    """
        this only made for the XOR 
    """
    input_list = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
    output_list = [0.0,1.0,1.0,0.0]
    fitness = 0 
    for i in range(10):
        value = randrange(len(input_list))
        output = phrnotype.activate(input_list[value])
        if round(output[0]) == output_list[value]:
            fitness += 1
        else:
            fitness -=1
    
    return fitness



def save_model(genome):
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    with open('models/model.pkl', 'wb') as file:
        pickle.dump(genome, file)