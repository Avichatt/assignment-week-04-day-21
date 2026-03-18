import numpy as np

# Part A: Concept Application

# 1. Create NumPy arrays of different dimensions
# 1D Array
arr_1d = np.array([1, 2, 3, 4, 5, 6])
print("1D Array:", arr_1d)

# 2D Array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D Array:\n", arr_2d)

# 3D Array
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("3D Array:\n", arr_3d)

# Indexing and Slicing
print("Element at [1, 2] in 2D array:", arr_2d[1, 2])
print("First row of 2D array:", arr_2d[0, :])
print("Second column of 2D array:", arr_2d[:, 1])
print("Subarray (2x2) from 2D array:\n", arr_2d[0:2, 0:2])

# 2. Basic operations without loops
a = np.array([10, 20, 30])
b = np.array([1, 2, 3])

print("Element-wise addition:", a + b)
print("Element-wise subtraction:", a - b)
print("Element-wise multiplication:", a * b)

# Mean, Variance, and Standard Deviation
data = np.array([1, 5, 10, 15, 20])
print("Mean:", np.mean(data))
print("Variance:", np.var(data))
print("Standard Deviation:", np.std(data))

# 3. Demonstration of Broadcasting
# Add a 1D array to a 2D array
ones_2d = np.ones((3, 3))
row_to_add = np.array([1, 2, 3])
broadcast_sum = ones_2d + row_to_add
print("2D array + 1D array (Broadcasting):\n", broadcast_sum)
# Explanation: NumPy expands the 1D array to match the shape of the 2D array for each row.

# Multiply a matrix (2D) by a scalar
scalar = 10
print("Matrix * Scalar:\n", arr_2d * scalar)
# Explanation: The scalar is multiplied by every element in the matrix.

# Multiply a matrix by a vector
vector = np.array([1, 0, 1])
print("Matrix * Vector (Broadcasting):\n", arr_2d * vector)
# Explanation: Each row of the matrix is multiplied element-wise by the vector.

# 4. Vectorized Operations
# Square and cube
nums = np.array([1, 2, 3, 4])
print("Squares:", np.square(nums))
print("Cubes:", nums**3)

# Replace negative values with 0
mixed_nums = np.array([-5, 10, -2, 8, 0])
mixed_nums[mixed_nums < 0] = 0
print("Negatives replaced with 0:", mixed_nums)

# Normalize an array (scale values between 0 and 1)
# Formula: (x - min) / (max - min)
to_normalize = np.array([10, 20, 30, 40, 50])
min_val = np.min(to_normalize)
max_val = np.max(to_normalize)
normalized = (to_normalize - min_val) / (max_val - min_val)
print("Normalized Array:", normalized)

# 5. Dataset Operations (NumPy array)
dataset = np.array([15, 2, 88, 44, 102, 7, 56, 91])
# Find top 5 maximum values
top_5 = np.sort(dataset)[-5:]
print("Top 5 max values:", top_5[::-1]) # Descending order

# Compute row-wise and column-wise sums
matrix_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Row-wise sums:", np.sum(matrix_data, axis=1))
print("Column-wise sums:", np.sum(matrix_data, axis=0))

# Identify indices of values greater than a threshold
threshold = 5
indices = np.where(matrix_data > threshold)
print("Indices where values > 5:", list(zip(indices[0], indices[1])))
