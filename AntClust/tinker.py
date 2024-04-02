import math
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def scaled_sigmoid(x, a, b):
    sigmoid_output = sigmoid(x)
    scaled_output = (b - a) * sigmoid_output + a
    return scaled_output

val = []
x = 0.5

for i in range(100):
    val.append(x*scaled_sigmoid(i, 0, 1.2))

plt.plot(range(100), val)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of Values against Index')
plt.grid(True)
plt.show()