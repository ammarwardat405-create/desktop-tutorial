# Adam - Momentum + RMSprop
# Linear Regression on Auto MPG
# pip install numpy matplotlib pandas scikit-learn ucimlrepo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo

np.random.seed(0)

# ---------- 1) Load data with pandas ----------
auto = fetch_ucirepo(id=9)
df = pd.concat([auto.data.features, auto.data.targets], axis=1)
df = df.dropna()                                   # remove missing values

X_df = df.drop(columns=["mpg"])                    # features
y_sr = df["mpg"]                                   # target

X_df = (X_df - X_df.mean()) / X_df.std()           # scale features
y_sr = (y_sr - y_sr.mean()) / y_sr.std()           # scale target
X_df.insert(0, "bias", 1.0)                        # column of 1s for the intercept

X = X_df.values
y = y_sr.values
n, d = X.shape

EPOCHS = 100


# ---------- 2) Loss (MSE) and gradient ----------
def loss(w):
    return np.mean((X @ w - y) ** 2)


def gradient(Xb, yb, w):
    return 2 / len(yb) * Xb.T @ (Xb @ w - yb)


# ---------- 3) Adam - Momentum + RMSprop ----------
lr = 0.01
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
batch_size = 32
w = np.zeros(d)
m = np.zeros(d)                                    # momentum part
s = np.zeros(d)                                    # RMSprop part
t = 0
losses = [loss(w)]

for epoch in range(EPOCHS):
    order = np.random.permutation(n)
    for i in range(0, n, batch_size):
        batch = order[i:i + batch_size]
        g = gradient(X[batch], y[batch], w)
        t = t + 1
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)               # bias correction
        s_hat = s / (1 - beta2 ** t)
        w = w - lr * m_hat / (np.sqrt(s_hat) + eps)
    losses.append(loss(w))

# ---------- 4) Compare with sklearn LinearRegression ----------
sk = LinearRegression().fit(X_df.drop(columns="bias"), y)
sk_mse = mean_squared_error(y, sk.predict(X_df.drop(columns="bias")))

print("Adam final MSE   :", round(losses[-1], 4))
print("sklearn reference MSE:", round(sk_mse, 4))
print("\nLearned weights:")
print(pd.DataFrame({"Adam": w, "sklearn": np.r_[sk.intercept_, sk.coef_]},
                   index=X_df.columns).round(3))

# ---------- 5) Plot ----------
plt.plot(losses, label="Adam")
plt.axhline(sk_mse, color="black", linestyle="--", label="sklearn")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Adam - Linear Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
