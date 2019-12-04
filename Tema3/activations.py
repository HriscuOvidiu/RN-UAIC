import numpy as np

def sigmoid(s, d=False):
    if d:
        return s * (1 - s)
    else:
        return 1/(1 + np.exp(-s))

def softmax(s, d=False):
    if d:
        return s
    else:
        exps = np.exp(s - np.max(s, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)