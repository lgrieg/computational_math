import numpy as np
import matplotlib.pyplot as plt
import time

def seidel(A, b, max_iter, tol):
    n = len(b)
    x = np.zeros(n)
    residuals = []
    
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        
        residual = np.linalg.norm(x_new - x)
        residuals.append(residual)
        if residual < tol:
            break
        x = x_new
    return x, residuals


n = 100
A = np.random.rand(n, n)
A = np.dot(A, A.T)
b = np.random.rand(n)
    
start_time_exact = time.time()
exact_solution = np.linalg.solve(A, b)
end_time_exact = time.time()
    
start_time_seidel = time.time()
max_iter = 10000000
tol = 1e-6
solution, residuals = seidel(A, b, max_iter, tol)
end_time_seidel = time.time()
    
print("Number of iterations:", len(residuals))
print("Exact error:", np.linalg.norm(solution - exact_solution))

print("Time taken for exact solution:", end_time_exact - start_time_exact, "seconds")
print("Time taken for Seidel method:", end_time_seidel - start_time_seidel, "seconds")

#print('solution ', solution, 'exact solution ', exact_solution)
#print('residuals', residuals)

plt.plot(np.log(residuals))
plt.xlabel('Iteration')
plt.ylabel('Log Residual')
plt.title('Convergence of Seidel Method')
plt.show()
    

