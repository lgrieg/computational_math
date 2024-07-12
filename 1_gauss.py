import numpy as np

def gauss(A, b):
    n = len(b)
    P = np.eye(n)
    L = np.eye(n)
    U = A.copy()
    x = np.zeros(n)
    y = np.zeros(n)

    for k in range(n):
        max_row = np.argmax(np.abs(U[k:n, k])) + k
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]

    for k in range(n):
        for i in range(k+1, n):
            if U[i, k] != 0:
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] -= L[i, k] * U[k, k:]

    # Ly = Pb
    matrix = P @ b
    for i in range(n):
        y[i] = matrix[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]

    # Ux = y
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            print(j)
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
        print(x[i])

    return L, U, P, x

A = np.array([[2.0, 4.0, 5.0], [6.0, 3.0, 2.0], [7.0, 8.0, 9.0]])
b = np.array([25.0, 18.0, 50.0])

L, U, P, x = gauss(A, b)
print("L:")
print(L)
print("U:")
print(U)
print("P:")
print(P)
print("x:")
print(x)

x_numpy = np.linalg.solve(A, b)
print("Решение numpy")
print(x_numpy)
print("Норма разницы:")
print(np.linalg.norm(x - x_numpy))

n = 5
k = 15
diag_matrices = [np.random.rand(n, n) for _ in range(k)]
matrices = [np.random.rand(n, n) for _ in range(k)]
vectors = [np.random.rand(n) for _ in range(k)]

for i in range(k):
    for j in range(n):
        diag_matrices[i][j][j] += 100000

diag_norms = np.zeros(k)
norms = np.zeros(k)
for i in range(k):
    L, U, P, x = gauss(diag_matrices[i], vectors[i])
    x_numpy = np.linalg.solve(diag_matrices[i], vectors[i])
    print("Решение numpy")
    print(x_numpy)
    print("Норма разницы:")
    diag_norms[i] = np.linalg.norm(x - x_numpy)
    print(np.linalg.norm(x - x_numpy))

for i in range(k):
    L, U, P, x = gauss(matrices[i], vectors[i])
    print("x:")
    print(x)
    x_numpy = np.linalg.solve(matrices[i], vectors[i])
    norms[i] = np.linalg.norm(x - x_numpy)
    print(np.linalg.norm(x - x_numpy))

print(" ")
print("  ")
print("diag differences", diag_norms)
print("non diag differences", norms)