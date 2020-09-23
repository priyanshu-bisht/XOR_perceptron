import numpy as np


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_p(x):
    return (x)*(1-(x))
    

w1 = np.random.randn(2, 2)
b1 = np.random.randn(1, 2)
w2 = np.random.randn(2, 1)
b2 = np.random.randn(1, 1)

for r in range(10000):
     layer1 = sigmoid(X @ w1 + b1)
     ans = sigmoid(layer1 @ w2 + b2)
     err = y - ans
     
     dans = err * sigmoid_p(ans)
     errlayere1 = dans @ w2.T
     dlayer1 = errlayere1 * sigmoid_p(layer1)
     
     w2 += layer1.T @ dans * 0.1
     b2 += sum(dans) * 0.1
     w1 += x.T @ dlayer1 * 0.1
     b1 += sum(dlayer1) * 0.1
     
print(ans)
