# Part D - AI-Augmented Task

1. Prompt AI:
"Explain regression vs classification and bias-variance tradeoff with Python examples 
and visualizations."

2. AI Output (Generated):
"""
Explanation:

1. Regression vs Classification:
- Regression: Predicts a house price based on its square footage. 
  The target variable is a number.
- Classification: Predicts whether an email is spam or not spam. 
  The target variable is a label.

2. Bias-Variance Tradeoff:
- Bias: Error from overly simple assumptions (Underfitting). 
  A straight line to fit a curved parabola.
- Variance: Error from over-sensitivity to fluctuations (Overfitting). 
  A complex curve that connects every point in the training set.

Python Example (Pseudo-code for visualization):
import matplotlib.pyplot as plt
import numpy as np

# Regression (Line)
x = np.linspace(0, 10, 10)
y = 2*x + 1
plt.scatter(x, y + np.random.randn(10))
plt.plot(x, y, label='Regression')

# Classification (Decision boundary)
c1 = np.random.randn(10) + 2
c2 = np.random.randn(10) + 5
plt.scatter(c1, c1, color='red')
plt.scatter(c2, c2, color='blue')
# Bias-Variance plots often show U-curves of training/test error.
"""

3. Evaluation:
Are explanations correct?
Answer: Yes. The explanation clearly distinguishes target variable types 
(labels vs numbers) and correctly identifies Bias as Underfitting and 
Variance as Overfitting.

Do visualizations correctly show underfitting and overfitting?
Answer: The pseudo-code description suggests using a straight line for 
regression and a scatter plot for classification. While it doesn't 
provide a full implementation, the concept of showing the U-curve 
for Bias-Variance is mentioned and is correct for visualizing the tradeoff.
