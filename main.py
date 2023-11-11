import numpy as np
import matplotlib.pyplot as plt

# Função de ativação e sua derivada

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

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
print(X)

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])
print(Y)

# Taxa de aprendizagem

N = 0.5

# Quantidade de épocas

EPOCHS = 10000

# Vetor da função de custo

cost = np.array([])

# Arquitetura da rede

n_neurons_input_layer = 2
n_neurons_hidden_layer_1 = 5
n_neurons_hidden_layer_2 = 3
n_neurons_output_layer = 1

# Pesos

w_hidden_layer_1 = np.random.rand(n_neurons_input_layer, n_neurons_hidden_layer_1)
print(w_hidden_layer_1)

w_hidden_layer_2 = np.random.rand(n_neurons_hidden_layer_1, n_neurons_hidden_layer_2)
print(w_hidden_layer_2)

w_output_layer = np.random.rand(n_neurons_hidden_layer_2, n_neurons_output_layer)
print(w_output_layer)

# Vieses

b_hidden_layer_1 = np.zeros(n_neurons_hidden_layer_1)
print(b_hidden_layer_1)

b_hidden_layer_2 = np.zeros(n_neurons_hidden_layer_2)
print(b_hidden_layer_2)

b_output_layer = np.zeros(n_neurons_output_layer)
print(b_output_layer)

# Treino da rede

for epoch in range(EPOCHS):
    activation_hidden_layer_1 = sigmoid(np.dot(X, w_hidden_layer_1) + b_hidden_layer_1)
    activation_hidden_layer_2 = sigmoid(np.dot(activation_hidden_layer_1, w_hidden_layer_2) + b_hidden_layer_2)
    activation_output_layer = sigmoid(np.dot(activation_hidden_layer_2, w_output_layer) + b_output_layer)
    
    cost = np.append(cost, MSE(Y, activation_output_layer))
    
    delta_output_layer = (Y - activation_output_layer) * sigmoid_derivative(activation_output_layer)
    delta_hidden_layer_2 = np.dot(delta_output_layer, w_output_layer.T) * sigmoid_derivative(activation_hidden_layer_2)
    delta_hidden_layer_1 = np.dot(delta_hidden_layer_2, w_hidden_layer_2.T) * sigmoid_derivative(activation_hidden_layer_1)
    
    w_output_layer += N * np.dot(activation_hidden_layer_2.T, delta_output_layer)
    w_hidden_layer_2 += N * np.dot(activation_hidden_layer_1.T, delta_hidden_layer_2)
    w_hidden_layer_1 += N * np.dot(X.T, delta_hidden_layer_1)

# Gráfico da função de custo

plt.plot(cost)
plt.title('Função de custo da rede')
plt.xlabel('Épocas')
plt.ylabel('Custo')
plt.show()

