import numpy as np
import func as f

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = f.softmax(z)
        loss = f.cross_entropy_error(y, t)

        return loss