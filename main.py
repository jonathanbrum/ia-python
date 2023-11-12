import operator as op
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

n_neurons_input_layer = X[0].size
n_neurons_hidden_layer = [5,3]
n_neurons_output_layer = Y[0].size

# Pesos
index = 0
w_hidden_layer = []
w_hidden_layer.append(np.random.rand(n_neurons_input_layer, n_neurons_hidden_layer[index]))
for neurons_hidden_layer in n_neurons_hidden_layer:
    if index == 0:
        index += 1
        continue
    w_hidden_layer.append(np.random.rand(n_neurons_hidden_layer[index - 1], neurons_hidden_layer))
    index += 1
print('w_hidden_layer',w_hidden_layer)
w_output_layer = np.random.rand(n_neurons_hidden_layer[index - 1], n_neurons_output_layer)
print('w_output_layer',w_output_layer)

# Vieses
index = 0
b_hidden_layer = []
for neurons_hidden_layer in n_neurons_hidden_layer:
    b_hidden_layer.append(np.zeros(neurons_hidden_layer))
    print('b_hidden_layer[',index,']: ',b_hidden_layer[index])
    index = index + 1
b_output_layer = np.zeros(n_neurons_output_layer)
print('b_output_layer',b_output_layer)

# Treino da rede
i = 0
_cost = 1
# delta_hidden_layer = np.zeros(len(n_neurons_hidden_layer))
delta_hidden_layer = np.zeros((4,3))
while ((_cost > 0.0005) & (i <= EPOCHS)):
    index = 0
    activation_hidden_layer = []
    activation_hidden_layer.append(np.dot(X, w_hidden_layer[index]) + b_hidden_layer[index])
    for neurons_hidden_layer in n_neurons_hidden_layer:
        if index == 0:
            index += 1
            continue
        activation_hidden_layer.append(np.dot(sigmoid(activation_hidden_layer[index - 1]), w_hidden_layer[index]) + b_hidden_layer[index])
        index += 1
    activation_output_layer = np.dot(sigmoid(activation_hidden_layer[index - 1]), w_output_layer) + b_output_layer
    
    _cost = MSE(Y, sigmoid(activation_output_layer))
    cost = np.append(cost, _cost)

    index = len(n_neurons_hidden_layer) - 1
    delta_output_layer = (Y - sigmoid(activation_output_layer)) * sigmoid_derivative(activation_output_layer)
    print(delta_hidden_layer,index)
    print(delta_hidden_layer[index],np.dot(delta_output_layer, w_output_layer.T) * sigmoid_derivative(activation_hidden_layer[index]))
    delta_hidden_layer[index] = np.dot(delta_output_layer, w_output_layer.T) * sigmoid_derivative(activation_hidden_layer[index])
    
    for neurons_hidden_layer in n_neurons_hidden_layer:
        if index == len(n_neurons_hidden_layer)- 1:
            continue
        delta_hidden_layer.insert(0,np.dot(delta_hidden_layer[index + 1], w_hidden_layer[index + 1].T) * sigmoid_derivative(activation_hidden_layer[index]))
        index -= 1    

    index = len(n_neurons_hidden_layer) - 1
    w_output_layer += N * np.dot(sigmoid(activation_hidden_layer[index]).T, delta_output_layer)
    print(w_hidden_layer[index])
    print(sigmoid(activation_hidden_layer[index - 1]).T)
    print(delta_hidden_layer,index)
    print(delta_hidden_layer[index])
    print(N * np.dot(sigmoid(activation_hidden_layer[index - 1]).T, delta_hidden_layer[index]))
    w_hidden_layer[index] += N * np.dot(sigmoid(activation_hidden_layer[index - 1]).T, delta_hidden_layer[index])

    for neurons_hidden_layer in n_neurons_hidden_layer:
        if index == n_neurons_hidden_layer.size - 1:
            continue        
        w_hidden_layer[index] += N * np.dot(X.T, delta_hidden_layer[index])
        index -= 1    
    
    i += 1

print(w_hidden_layer)
print(w_output_layer)
print(b_hidden_layer)
print(b_output_layer)
# Gráfico da função de custo

plt.plot(cost)
plt.title('Função de custo da rede')
plt.xlabel('Épocas')
plt.ylabel('Custo')
plt.show()

