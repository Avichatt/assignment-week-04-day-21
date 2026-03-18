import numpy as np

# Part C - Interview Ready

# Q1 — What is the difference between regression and classification? 
# Give real-world examples.
# Answer: The main difference is the output type.
# - Regression predicts a continuous numerical quantity (e.g., predicting 
#   the exact temperature for tomorrow).
# - Classification predicts a categorical label (e.g., predicting 
#   whether it will be 'hot' or 'cold' tomorrow).

# Q2 (Coding) — Implement a function: calculate_mse(y_true, y_pred)
def calculate_mse(y_true, y_pred):
    # Mean Squared Error: mean((y_true - y_pred)**2)
    # y_true and y_pred should be numpy arrays
    diff = np.array(y_true) - np.array(y_pred)
    mse = np.mean(diff**2)
    return mse

# Test calculation
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
result = calculate_mse(y_true, y_pred)
print(f"Computed MSE: {result}") # Expected output: 0.375

# Q3 — Explain bias–variance tradeoff. What happens in:
# ● Underfitting: High Bias/Low Variance. The model is too simple 
#   and fails to capture any pattern from the training data, 
#   resulting in poor performance on both training and test data.
# ● Overfitting: Low Bias/High Variance. The model is too complex 
#   and learns the training noise along with the signal, 
#   meaning it performs great on training data but poorly on test data.
# ● The "Tradeoff": As you decrease bias (make model more complex), 
#   variance increases (model becomes more sensitive to noise). 
#   The goal is to find the "sweet spot" at minimum total error.
