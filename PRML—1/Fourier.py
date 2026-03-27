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

# 傅里叶拟合线性回归（GD法）
class FourierRegressionGD:
    def __init__(self, n_terms=5, period=None, lr=0.01, epochs=1000):
        self.n_terms = n_terms
        self.period = period
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.intercept = 0.0
        self.loss_history = []

    def _build_features(self, X):
        if self.period is None:
            self.period = X.max() - X.min()
        X_norm = 2 * np.pi * X / self.period
        features = [np.ones(X.shape[0])]
        for k in range(1, self.n_terms + 1):
            features.append(np.sin(k * X_norm))
            features.append(np.cos(k * X_norm))
        return np.column_stack(features)

    def fit(self, X, y):
        X_feat = self._build_features(X)
        theta = np.zeros(X_feat.shape[1])
        for _ in range(self.epochs):
            y_pred = X_feat @ theta
            error = y_pred - y
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)
            grad = (2 / len(y)) * (X_feat.T @ error)
            theta -= self.lr * grad
        self.intercept = theta[0]
        self.weights = theta[1:]
        return self

    def predict(self, X):
        X_feat = self._build_features(X)
        return X_feat @ np.concatenate([[self.intercept], self.weights])

# 训练与评估
model = FourierRegressionGD(n_terms=5, lr=0.01, epochs=5000).fit(X_train, y_train)
mse = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)

print(f"Training MSE: {mse(y_train, model.predict(X_train)):.4f}")
print(f"Testing MSE: {mse(y_test, model.predict(X_test)):.4f}")
print(f"Intercept: {model.intercept:.4f}")
print(f"Weights: {model.weights.round(4)}")

# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, label='Training Data')
X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, 'r-', linewidth=2, label='Fourier GD Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.6, label='Testing Data')
X_line_test = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_line_test = model.predict(X_line_test)
plt.plot(X_line_test, y_line_test, 'r-', linewidth=2, label='Fourier GD Fitted Line')
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
plt.scatter(sample_epochs, sample_losses, color='red', s=8, label='Sample Every 10 Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()