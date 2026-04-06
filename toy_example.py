import numpy as np

X = np.array([0.0, 1.0, 0.5, 0.2])
y = np.array([0.0, 1.0, 0.5, 0.2])

def h(x, w1, w2):
    return w1 * x + w2 * x**2

def loss(w1, w2):
    preds = h(X, w1, w2)
    return np.sum((preds - y) ** 2)

def gradients(w1, w2):
    preds = h(X, w1, w2)
    errors = preds - y
    dL_dw1 = 2 * np.sum(errors * X)
    dL_dw2 = 2 * np.sum(errors * X**2)
    return dL_dw1, dL_dw2

w1, w2 = 143.0, 13.0
lr = 0.1

for step in range(10000):
    dL_dw1, dL_dw2 = gradients(w1, w2)
    w1 -= lr * dL_dw1
    w2 -= lr * dL_dw2

print("Learned weights:")
print(f"w1 = {w1:.4f}, w2 = {w2:.4f}")

print("At x = 0.5:")
print(f"h(0.5) = {h(0.5, w1, w2):.4f}")

w1_nice, w2_nice = w1, w2
w1_bad, w2_bad = -10.0, 11.0

print("Manual solutions:")
print(f"nice: loss={loss(w1_nice, w2_nice):.4f}, h(0.5)={h(0.5, w1_nice, w2_nice):.4f}")
print(f"bad:  loss={loss(w1_bad, w2_bad):.4f}, h(0.5)={h(0.5, w1_bad, w2_bad):.4f}")