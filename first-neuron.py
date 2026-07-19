Progressive code for my first neuron, including inputs, weights, bias and activation functions
First code :-
# x1 = 5
# x2 = 3
# x3 = 8
# w1 = 0.5
# w2 = -0.2
# w3 = 1.1
# b = 0.5
# z = (x1 * w1) + (x2 * w2) + (x3 * w3) + b
# print(f"Output: {z:2f}")

Second improved:- 
# print("Enter the number of inputs:")
# n = int(input())
# inputs = []
# weights = []
# b = 0.5
# z = 0
# for i in range(n):
#     print("Enter input", i + 1, ":")
#     inputs.append(float(input()))
#     print("Enter weight", i + 1, ":")
#     weights.append(float(input()))
# for i in range(n):
#     z += (inputs[i] * weights[i])
# z += b
# print(f"Output: {z:2f}")

Final neuron:-
import math

print("Enter the number of inputs:")
n = int(input())

inputs = []
weights = []
bias = 0.5
for i in range(n):
    print("Enter input", i + 1, ":")
    inputs.append(float(input()))
    print("Enter weight", i + 1, ":")
    weights.append(float(input()))

def weighted_sum(inputs, weights, bias):
    return sum(inputs[i] * weights[i] for i in range(n)) + bias

def step(x):
    return 1 if x > 0 else 0
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def relu(x):
    return max(0, x)
def tanh(x):
    return math.tanh(x)
def leaky_relu(x):
    return x if x > 0 else 0.01 * x
def gelu(x):
    return x * 0.5 * (1 + math.tanh(0.795 * x * (1 + 0.044715 * x * x)))
activation_functions = [step, sigmoid, relu, tanh, leaky_relu, gelu]

def test_all_activation(z_value):
    for activation_function in activation_functions:
        print("="*30)
        print(activation_function.__name__ + " function:")
        output = activation_function(z_value)
        print(f"{output:.2f}")
        print("="*30)
print("Without bias:")
test_all_activation(weighted_sum(inputs, weights, 0))

print("Zero weights:")
zeroweights = weights.copy()
for i in range(n):
    zeroweights[i] = 0
test_all_activation(weighted_sum(inputs, zeroweights, bias))
