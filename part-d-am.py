# Part D - AI-Augmented Task

# 1. Prompt AI:
# "Explain NumPy broadcasting and vectorisation with practical Python examples."

# 2. AI Output (Generated):
"""
Explanation:

1. NumPy Vectorisation:
Vectorisation allows you to perform operations on an entire array at once 
rather than iterating through each element using a loop. 
This is much faster because NumPy iterates through the elements in 
optimized C/C++ code.

Example:
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = a * b # Vectorised element-wise multiplication
print(result) # [4 10 18]

2. NumPy Broadcasting:
Broadcasting allows operations to occur between arrays of different 
shapes. NumPy automatically stretches the smaller array to match 
the shape of the larger array for the operation.

Example:
ones = np.ones((2, 3)) # (2, 3) 2D array
ones_1d = np.array([1, 2, 3]) # (3,) 1D array
broadcast_sum = ones + ones_1d # stretches ones_1d to [[1,2,3], [1,2,3]]
print(broadcast_sum) 
# [[2. 3. 4.], [2. 3. 4.]]
"""

# 3. Evaluation:
# Are examples correct?
# Answer: Yes, the examples correctly demonstrate both vectorisation 
# and broadcasting (1D being added to 2D).

# Is code efficient and runnable?
# Answer: Yes, the code uses standard NumPy and is efficient. 
# It can be run directly using Python and will give the expected results.
