from neat.population import Population
from neat.network import Network
import pickle

def train():
    p = Population()
    p.evolve(count=10, noGeneration=80, noInput=2, noOutput=1)  


def load_model():
    with open('models/model.pkl', 'rb') as f:
        genome = pickle.load(f)

    network = Network(genome=genome)
    output = network.activate([1,0])
    print(output)

if __name__ == "__main__":
    train()
    load_model()