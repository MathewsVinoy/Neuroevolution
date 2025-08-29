import numpy as np

class Linear:
    def __init__(self, weight, bias):
        self.W = np.array(weight, dtype=np.float32)
        self.B = np.array(bias, dtype=np.float32)

    def forward(self, x):
        x = np.array(x, dtype=np.float32)
        return self.B + np.dot(self.W.T, x)