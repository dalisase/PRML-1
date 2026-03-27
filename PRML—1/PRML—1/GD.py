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

# 梯度下降法线性回归
class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.intercept = 0.0
        self.weights = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0.0

        for _ in range(self.epochs):
            y_pred = X @ self.weights + self.intercept
            error = y_pred - y

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.intercept -= self.lr * db
            self.loss_history.append(np.mean(error**2))

        return self

    def predict(self, X):
        return X @ self.weights + self.intercept

# 训练与评估
model = LinearRegressionGD(lr=0.01, epochs=5000).fit(X_train, y_train)
mse = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)

print(f"Training MSE: {mse(y_train, model.predict(X_train)):.4f}")
print(f"Testing MSE: {mse(y_test, model.predict(X_test)):.4f}")
print(f"Intercept: {model.intercept:.4f}")
print(f"Slope: {model.weights.round(4)}")


checkpoints = [10, 50, 100, 200, 500, 600, 700, 800, 900,1000,2000, 3000, 4000, 5000]
for n in checkpoints:
    if n <= len(model.loss_history):
        print(f"epoch {n}: loss={model.loss_history[n-1]:.6f}")
    else:
        print(f"epoch {n}: not reached (epochs={len(model.loss_history)})")

# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, label='Training Data')
X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, 'r-', linewidth=2, label='GD Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.6, label='Testing Data')
X_line_test = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_line_test = model.predict(X_line_test)
plt.plot(X_line_test, y_line_test, 'r-', linewidth=2, label='GD Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Testing')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Loss 曲线（每10个epoch取点）
epochs = len(model.loss_history)
sample_epochs = list(range(10, epochs + 1, 10))
sample_losses = [model.loss_history[i - 1] for i in sample_epochs]

plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, epochs + 1), model.loss_history, color='blue', linewidth=1, label='Loss')
plt.scatter(sample_epochs, sample_losses, color='red', s=2, label='Sample Every 10 Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()