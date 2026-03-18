import numpy as np
import time

# Part B - Stretch Problem

# 1. Matrix operations
# Matrix Multiplication
# A (2x3) * B (3x2) = C (2x2)
A = np.array([[10, 20, 30], [5, 10, 15]])
B = np.array([[1, 2], [3, 4], [5, 6]])
# Matrix Multiplication using np.dot() or @
C = A @ B
print("Matrix Multiplication (A @ B):\n", C)

# Transpose
# Transpose of A (2x3) becomes (3x2)
print("A transposed:\n", A.T)

# Determinant
# Determinant must be of a square matrix
D = np.array([[3, 8], [4, 6]])
# Determinant using np.linalg.det()
det_D = np.linalg.det(D)
print("Determinant of D:", det_D)

# 2. Solving a system of linear equations
# 3x + 2y = 8
# 1x + 2y = 4
# Matrix representation: Ax = B
# A = [[3, 2], [1, 2]]
# B = [8, 4]
system_A = np.array([[3, 2], [1, 2]])
system_B = np.array([8, 4])
# Solving using np.linalg.solve()
solution = np.linalg.solve(system_A, system_B)
print("Solution for 3x+2y=8 and 1x+2y=4: x =", solution[0], ", y =", solution[1])

# 3. Performance Comparison
# Large array (10 million elements)
size = 10_000_000
large_array = np.random.rand(size)

# Python loop to sum elements
start_time_loop = time.time()
sum_loop = 0
for val in large_array:
    sum_loop += val
end_time_loop = time.time()
time_loop = end_time_loop - start_time_loop
print("Python loop time:", time_loop, "seconds")

# NumPy vectorized sum
start_time_numpy = time.time()
sum_numpy = np.sum(large_array)
end_time_numpy = time.time()
time_numpy = end_time_numpy - start_time_numpy
print("NumPy sum time:", time_numpy, "seconds")

# Report time difference and explanation
diff = time_loop / time_numpy
print("NumPy was approximately", round(diff, 2), "times faster!")
# Explanation: NumPy is faster because it's written in optimized C/C++ 
# and uses vectorized operations that can process large blocks of memory 
# at once, whereas Python loops have high overhead for each iteration.
