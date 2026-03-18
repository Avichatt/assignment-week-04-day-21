import numpy as np
import matplotlib.pyplot as plt

# Part B - Bias-Variance Tradeoff (Pure NumPy Version)

# 1. Simulate bias–variance tradeoff
# Synthetic Sine Wave Dataset
X = np.sort(np.random.rand(20) * 10) # 20 points from 0 to 10
y = np.sin(X) + np.random.normal(0, 0.2, len(X)) # Sine wave + noise

# Test data (More samples for smooth curve and evaluation)
X_test = np.linspace(0, 10, 100)
y_test = np.sin(X_test)

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Plot training vs test data
plt.figure(figsize=(15, 5))

# Fit models of increasing complexity (Degree 1, 15)
degrees = [1, 15]

for i, degree in enumerate(degrees):
    # Fit the polynomial model (mx + c for degree 1, high complexity for 15)
    coeffs = np.polyfit(X, y, degree)
    p = np.poly1d(coeffs)
    
    plt.subplot(1, 4, i + 1)
    plt.scatter(X, y, color='blue', label='Train Data (Noise)')
    plt.plot(X_test, p(X_test), color='red', label='Model Fit')
    plt.title(f"Degree {degree} ({'Underfit' if degree == 1 else 'Overfit'})")
    plt.legend()

# 2. Plot Training Error vs Model Complexity
complexity = list(range(1, 11)) # Testing degrees 1 to 10
train_errors = []
test_errors = []

for degree in complexity:
    coeffs = np.polyfit(X, y, degree)
    p = np.poly1d(coeffs)
    
    train_errors.append(calculate_mse(y, p(X)))
    test_errors.append(calculate_mse(y_test, p(X_test)))

# Visualization of Error Tradeoff
plt.subplot(1, 2, 2)
plt.plot(complexity, train_errors, label='Training Error (Bias)', color='green')
plt.plot(complexity, test_errors, label='Testing Error (Generalization)', color='orange')
plt.axvline(x=4, color='gray', linestyle=':', label='Sweet Spot') # Approximation
plt.title("Tradeoff: Training vs Testing Error")
plt.xlabel("Model Complexity (Degree)")
plt.ylabel("MSE")
plt.legend()

# 3. Explanations
# Bias: High bias means the model is too simple (Degree 1). 
# It can't capture the pattern.
# Variance: High variance means the model is too complex (Degree 15). 
# It connects all the dots of noise.
# Optimal Model: Minimal testing error. This is where we balance 
# simplicity and flexibility.

plt.savefig('bias_variance_plots_pm.png')
print("Bias-Variance plots saved to bias_variance_plots_pm.png")
