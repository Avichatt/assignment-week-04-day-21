import numpy as np
import matplotlib.pyplot as plt

# Part A - Concept Application (Pure NumPy Version)

# 1. Create synthetic datasets
# Regression Dataset (Continuous target)
# y = 2x + 5 + some noise
X_reg = np.linspace(0, 10, 50)
y_reg = 2 * X_reg + 5 + np.random.randn(50) * 2

# Classification Dataset (Binary target)
# Clusters around 2 and 8
X_cls = np.concatenate([np.random.normal(2, 1, 25), np.random.normal(8, 1, 25)])
y_cls = np.concatenate([np.zeros(25), np.ones(25)])

# 2. Train models and visualize
# Simple Linear Regression (y = mx + c) using np.polyfit
m, c = np.polyfit(X_reg, y_reg, 1)
y_reg_pred = m * X_reg + c

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_reg, y_reg, color='blue', label='Actual Data')
plt.plot(X_reg, y_reg_pred, color='red', label=f'Model (y={m:.2f}x + {c:.2f})')
plt.title("Regression: Continuous Price Prediction")
plt.legend()

# Simple Classification Model (Decision Boundary at x=5)
# Logistic regression is more complex to write manually, so we'll use a 
# simple threshold classifier as 'basic logic'.
threshold = 5.0
y_cls_pred = (X_cls > threshold).astype(int)

plt.subplot(1, 2, 2)
plt.scatter(X_cls, np.random.rand(50), c=y_cls, cmap='bwr', label='Data')
plt.axvline(x=threshold, color='green', linestyle='--', label=f'Threshold={threshold}')
plt.title("Classification: Binary (Target 0 or 1)")
plt.legend()
plt.savefig('datasets_plots_pm.png')
print("Synthetic datasets plotted and saved to datasets_plots_pm.png")

# 3. Decision Logic: Identify problem type
# Dataset Example: Predicting if an animal is a Cat (0) or Dog (1) 
# -> Classification. Target is a label/category.
# Dataset Example: Predicting the exact weight of cattle (e.g., 500.5kg) 
# -> Regression. Target is a continuous number.

# 4. Manual Regression implementation
def manual_regression_mse():
    # True values vs Predictions
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])
    # MSE = mean((y_true - y_pred)**2)
    mse = np.mean((y_true - y_pred)**2)
    print("\nManual Regression:")
    print("MSE:", mse)

# 5. Manual Classification implementation
def manual_classification_accuracy():
    # True labels vs Predictions
    # Correct = 2, Total = 3 -> Accuracy = 0.66
    labels_true = np.array([0, 1, 1])
    labels_pred = np.array([0, 0, 1])
    accuracy = np.mean(labels_true == labels_pred)
    print("\nManual Classification:")
    print("Accuracy:", accuracy)

manual_regression_mse()
manual_classification_accuracy()
