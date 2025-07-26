from neat.network import Network
# from graphviz import Digraph
import pickle


def load_model():
    with open('models/model.pkl', 'rb') as f:
        genome = pickle.load(f)

    network = Network(genome=genome)
    output = network.activate([0, 0])
    print(output, round(output[0]))

    # dot = Digraph()
    # for node in network.nodes:
    #     dot.node(str(node.id), str(node.id))
    # for conn in network.conn:
    #     if conn.enable:
    #         dot.edge(str(conn.inId), str(conn.outId))

    # dot.render('test/nn', format='png', view=True)


load_model()
