import numpy as np
import matplotlib.pyplot as plt

def get_weights():
    x = [-1, -1/3, 1/3, 1]
    weight_dif = [1, 1, 1, 1]
    weights = [1, 1, 1, 1]
    for i in range(4):
        for j in range(4):
            if j != i:
                weight_dif[i] /= x[i] - x[j]
 #сложно посчитать функцию - посчитала интегралы весов руками (в вольфраме)

def composite_quadrature(f, a, b, tol):
    
    def integrate_segment(f, a, b):
        weights = [1/4, 3/4, 3/4, 1/4]
        x_interval = (a - b) / 3
        #assert(a + 3 * x_interval == b)
        x = [a, a + x_interval, a + 2 * x_interval, b]
        integral = (b-a)/2 * sum(weights[i] * f(x[i]) for i in range(4))
        return integral

    n = 1
    error = tol + 1
    integral_old = integrate_segment(f, a, b)
    
    while error > tol:
        n *= 2
        h = (b - a) / n
        integral_new = sum([integrate_segment(f, a + i*h, a + (i+1)*h) for i in range(n)])
        error = abs(integral_new - integral_old) / 15
        #print("old integral ", integral_old, " new integral ", integral_new, " error ", error)
        integral_old = integral_new
    
    return integral_old, n

def smooth_func(x):
    return x**2 + np.sin(x)

tolerances = [2**(-i) for i in range(5, 25, 5)]
segments = []

for tol in tolerances:
    integral, num_segments = composite_quadrature(smooth_func, -1, 1, tol)
    segments.append(num_segments)
    print("integral value is ", integral, " with blocks ", num_segments)

plt.figure(figsize=(10, 6))
plt.loglog(tolerances, segments, marker='o')
plt.xlabel('Tolerance')
plt.ylabel('Num of Segments')
plt.title('Segments vs Tolerance in Log Scale')
plt.grid(True)
plt.show()
