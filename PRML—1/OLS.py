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

# 最小二乘法
class LinearRegression:
    def fit(self, X, y):
        X_b = np.c_[np.ones(X.shape[0]), X]
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept, self.weights = theta[0], theta[1:]
        return self

    def predict(self, X):
        return X @ self.weights + self.intercept

# 训练和评估
model = LinearRegression().fit(X_train, y_train)
mse = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)

print(f"训练误差: {mse(y_train, model.predict(X_train)):.4f}")
print(f"测试误差: {mse(y_test, model.predict(X_test)):.4f}")
print(f"截距: {model.intercept:.4f}")
print(f"斜率: {model.weights.round(4)}")

# 绘图
plt.figure(figsize=(12, 5))

# 训练集
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, label='Training Data')
X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, 'r-', linewidth=2, label='Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training')
plt.legend()
plt.grid(True, alpha=0.3)

# 测试集
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.6, label='Testing Data')
X_line_test = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_line_test = model.predict(X_line_test)
plt.plot(X_line_test, y_line_test, 'r-', linewidth=2, label='Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Testing')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()