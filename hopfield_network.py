import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))

    # memorize
    def train(self, patterns):
        for p in patterns:
            p = np.array(p)
            self.W += np.outer(p, p)

        # delete self-connection
        np.fill_diagonal(self.W, 0)

        # normalize
        self.W /= len(patterns)

    # recall
    def recall(self, pattern, steps=10):
        x = np.array(pattern)

        for _ in range(steps):
            for i in range(self.size):
                raw = np.dot(self.W[i], x)
                x[i] = 1 if raw >= 0 else -1

        return x

if __name__ == "__main__":
    # item to memorize
    A = [1, -1, 1, -1]
    B = [-1, 1, -1, 1]

    net = HopfieldNetwork(size=4)
    net.train([A, B])

    # broken input
    noisy = [1, -1, 1, 1]

    # recall
    result = net.recall(noisy)

    print("input: ", noisy)
    print("recall: ", result.tolist())
