import numpy as np

# Toy batch of activations (batch_size=4, features=3)
X = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0],
    [3.0, 6.0, 9.0],
    [4.0, 8.0, 12.0]
])

epsilon = 1e-5

# Learnable parameters (normally trained during backprop)
gamma = np.ones(X.shape[1])   # scale
beta = np.zeros(X.shape[1])   # shift

# Step 1: compute mean of batch (per feature)
mean = np.mean(X, axis=0)

# Step 2: compute variance of batch (per feature)
variance = np.var(X, axis=0)

# Step 3: normalize
X_hat = (X - mean) / np.sqrt(variance + epsilon)

# Step 4: scale and shift
Y = gamma * X_hat + beta

print("Input:")
print(X)

print("\nNormalized Output:")
print(Y)