def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def step_function(x):
    return np.array(x > 0, dtype=np.int)
