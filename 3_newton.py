import numpy as np
import matplotlib.pyplot as plt

def prod(x, x_values, j):
    res = 1
    for i in range(j):
        res *= x - x_values[i]
    return res

def divided_differences(x, y):
    n = len(x)
    f = np.zeros((n, n))
    f[:, 0] = y
    
    for j in range(1, n):
        for i in range(n-j):
            f[i, j] = (f[i+1, j-1] - f[i, j-1]) / (x[i+j] - x[i])
    
    return f[0, :]

def newton_interpolation(x_nodes, y_nodes, x_values):
    n = len(x_nodes)
    coeffs = divided_differences(x_nodes, y_nodes)
    result = np.zeros_like(x_values)
    derivative = np.zeros_like(x_values)
    
    for i, x in enumerate(x_values):
        result[i] = sum(coeffs[j] * prod(x, x_nodes, j) for j in range(n))
        derivative[i] = sum(coeffs[j] * sum(prod(x, x_nodes, j) / (x - x_nodes[k] + 0.000000000000000001) for k in range (j)) for j in range(n))
    
    return result, derivative

def plot_interpolation(x_nodes, func, title):
    x_values = np.linspace(min(x_nodes), max(x_nodes), 10000)
    y_values, derivative = newton_interpolation(x_nodes, func(x_nodes), x_values)

    print("Exact error:", np.linalg.norm(y_values - func(x_values)))
    print("derivative ", derivative)
    print("Derivative error:", np.linalg.norm(derivative - func(x_values, derivative=True)))
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_values, func(x_values), label='Original Function')
    plt.plot(x_values, y_values, label='Interpolating Polynomial')
    plt.scatter(x_nodes, func(x_nodes), color='red', label='Interpolation Nodes')
    plt.legend()
    plt.title(title + ' - Function and Interpolating Polynomial')
    
    plt.subplot(1, 2, 2)
    plt.plot(x_values, func(x_values, derivative=True), label="Original Function's Derivative")
    plt.plot(x_values, derivative, label="Interpolating Polynomial's Derivative")
    plt.legend()
    plt.title(title + ' - Derivatives')
    
    plt.show()

# Пример гладкой функции f
def smooth_func(x, derivative=False):
    if not derivative:
        return - x**15
    else:
        return - 15 * x**14

# Равномерная сетка
x_uniform = np.linspace(-1, 1, 10)
plot_interpolation(x_uniform, smooth_func, 'Uniform Grid')

# Чебышёвская сетка
x_chebyshev = np.cos(np.pi * (2*np.arange(10) + 1) / (2*10))
plot_interpolation(x_chebyshev, smooth_func, "Chebyshev's Grid")