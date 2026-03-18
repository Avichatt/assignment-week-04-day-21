import numpy as np

# Part C - Interview Ready

# Q1 — What is NumPy broadcasting? Why is it useful?
# Answer: Broadcasting is a NumPy feature that allows operations 
# between arrays of different shapes. It's useful because it 
# automatically stretches the smaller array to match the larger 
# one's dimensions, making the operation possible. This avoids 
# creating large copies of data, which saves memory and increases speed.

# Q2 (Coding) — Implement: normalize(X)
# Scale values between 0 and 1 using NumPy
def normalize_array(X):
    # Min-Max Scaling: Every x = (x - min) / (max - min)
    min_val = np.min(X)
    max_val = np.max(X)
    
    # Check if max - min is 0 to avoid division by zero
    if max_val - min_val == 0:
        return np.zeros_like(X) # Or handle based on requirements
    
    normalized_array = (X - min_val) / (max_val - min_val)
    return normalized_array

# Test the normalize function
test_array = np.array([10, 20, 30, 40, 50])
print("Normalized values:", normalize_array(test_array))

# Q3 — What is the difference between vectorisation and loops? Why is NumPy faster?
# Answer: Vectorisation means performing operations on entire arrays 
# at once instead of one element at a time like a loop. 
# NumPy is faster because its core is written in C/C++, 
# which avoids Python's slow loop overhead. It also utilizes 
# hardware optimizations like SIMD (Single Instruction, Multiple Data) 
# and processes memory in efficient blocks.
