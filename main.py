import numpy as np
import matplotlib.pyplot as plt

# Função de ativação e sua derivada

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Função de custo

def MSE(Y_target, Y_pred):
    return np.mean((Y_target - Y_pred) ** 2)

# Definindo o dataset

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# print(X)

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])
# print(Y)

# Taxa de aprendizagem

N = 0.5

# Quantidade de épocas

EPOCHS = 100000

# Vetor da função de custo

cost = np.array([])

# Arquitetura da rede

n_neurons_input_layer = X[0].size
# n_neurons_hidden_layer = [5,3]
n_neurons_hidden_layer = [3,2]
n_neurons_output_layer = Y[0].size

# Pesos
w_hidden_layer = list(range(len(n_neurons_hidden_layer)))
w_hidden_layer[0] = np.random.rand(n_neurons_input_layer, n_neurons_hidden_layer[0])
for index in range(len(n_neurons_hidden_layer)):
    if index == 0:
        continue
    w_hidden_layer[index] = np.random.rand(n_neurons_hidden_layer[index - 1], n_neurons_hidden_layer[index])
w_output_layer = np.random.rand(n_neurons_hidden_layer[index], n_neurons_output_layer)

# Vieses
b_hidden_layer = list(range(len(n_neurons_hidden_layer)))
for index in range(len(n_neurons_hidden_layer)):
    b_hidden_layer[index] = np.zeros(n_neurons_hidden_layer[index])
b_output_layer = np.zeros(n_neurons_output_layer)

# Treino da rede
i = 0
_cost = 1
delta_hidden_layer = list(range(len(n_neurons_hidden_layer)))
activation_hidden_layer = list(range(len(n_neurons_hidden_layer)))
last_hidden_n_index = len(n_neurons_hidden_layer) - 1
while ((_cost > 0.0005) & (i <= EPOCHS)):
    i += 1
    activation_hidden_layer[0] = np.dot(X, w_hidden_layer[0]) + b_hidden_layer[0]
    for index in range(len(n_neurons_hidden_layer)):
        if index == 0:
            continue
        activation_hidden_layer[index] = np.dot(sigmoid(activation_hidden_layer[index - 1]), w_hidden_layer[index]) + b_hidden_layer[index]
    activation_output_layer = np.dot(sigmoid(activation_hidden_layer[index]), w_output_layer) + b_output_layer
    
    _cost = MSE(Y, sigmoid(activation_output_layer))
    cost = np.append(cost, _cost)

    delta_output_layer = (Y - sigmoid(activation_output_layer)) * sigmoid_derivative(activation_output_layer)
    delta_hidden_layer[last_hidden_n_index] = np.dot(delta_output_layer, w_output_layer.T) * sigmoid_derivative(activation_hidden_layer[last_hidden_n_index])
    for index in range(len(n_neurons_hidden_layer) - 1,-1,-1):
        if index == len(n_neurons_hidden_layer) - 1:
            continue
        delta_hidden_layer[index] = np.dot(delta_hidden_layer[index + 1], w_hidden_layer[index + 1].T) * sigmoid_derivative(activation_hidden_layer[index])

    w_output_layer += N * np.dot(sigmoid(activation_hidden_layer[last_hidden_n_index]).T, delta_output_layer)
    for index in range(len(n_neurons_hidden_layer) - 1,0,-1):
        w_hidden_layer[index] += N * np.dot(sigmoid(activation_hidden_layer[index - 1]).T, delta_hidden_layer[index])
    w_hidden_layer[0] += N * np.dot(X.T, delta_hidden_layer[0])
    
    print('--------------------------------')
    print(w_hidden_layer[0])
    print(w_hidden_layer[1], end="\r")
    # print(w_output_layer)
    # print(b_hidden_layer, end="\r")
    # print(b_output_layer)

# Gráfico da função de custo

plt.plot(cost)
plt.title('Função de custo da rede')
plt.xlabel('Épocas')
plt.ylabel('Custo')
plt.show()

