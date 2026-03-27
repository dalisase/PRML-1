import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 读取数据
current_dir = os.path.dirname(os.path.abspath(__file__))
train_data = pd.read_excel(os.path.join(current_dir, 'DATA.xlsx'))
test_data = pd.read_excel(os.path.join(current_dir, 'TEST.xlsx'))

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

# 牛顿法线性回归
class LinearRegressionNewton:
    def __init__(self, max_iter=50, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.intercept = 0.0
        self.weights = None
        self.loss_history = []

    def fit(self, X, y):
        n, d = X.shape
        X_b = np.c_[np.ones((n, 1)), X]   # [1, x]
        theta = np.zeros(d + 1)

        for _ in range(self.max_iter):
            y_pred = X_b @ theta
            error = y_pred - y
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            grad = (2.0 / n) * (X_b.T @ error)
            H = (2.0 / n) * (X_b.T @ X_b)

            # 牛顿步：theta = theta - H^{-1} grad
            delta = np.linalg.pinv(H) @ grad
            theta_new = theta - delta

            if np.linalg.norm(theta_new - theta) < self.tol:
                theta = theta_new
                break
            theta = theta_new

        self.intercept = theta[0]
        self.weights = theta[1:]
        return self

    def predict(self, X):
        return X @ self.weights + self.intercept

# 训练与评估
model = LinearRegressionNewton(max_iter=100, tol=1e-8).fit(X_train, y_train)
mse = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)

print(f"Training MSE: {mse(y_train, model.predict(X_train)):.4f}")
print(f"Testing MSE: {mse(y_test, model.predict(X_test)):.4f}")
print(f"Intercept: {model.intercept:.4f}")
print(f"Slope: {model.weights.round(4)}")

# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, label='Training Data')
X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, 'r-', linewidth=2, label='Newton Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.6, label='Testing Data')
X_line_test = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_line_test = model.predict(X_line_test)
plt.plot(X_line_test, y_line_test, 'r-', linewidth=2, label='Newton Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Testing')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
