---
title: Manual implementation of a network
layout: single
---

### Implementation of a one-layer neural network that solves the [XOR](https://en.wikipedia.org/wiki/Exclusive_or) task using only numpy, roughly following [this](https://cs231n.github.io/optimization-2/) excellent blogpost by Andrej Karpathy. Shared work with [ppries](https://github.com/ppries).

---


~~~python
from __future__ import print_function, division
import numpy as np


def forward(X, W_X_yhat, b_X_yhat):
    # Forward pass.
    # (N x D) * (D x K) = (N x K)
    logits = np.dot(X, W_X_yhat) + b_X_yhat
    yhat = 1 / (1 + np.exp(-logits))
    return yhat


def backward(X, y, yhat):
    # Backward pass. Unless otherwise indicated by the variable names,
    # all derivatives are of the cost function (cross entropy). E.g.
    # `dlogits` is really dcost_dlogits.

    # (N x K) - (N x K) = (N x K)
    dlogits = yhat - y
    # (H x N) * (N x K) = (H x N)
    dW_X_yhat = np.dot(X.T, dlogits)
    db_X_yhat = np.sum(dlogits, axis=0, keepdims=True)
    return dW_X_yhat, db_X_yhat


def eval(y, yhat):
    xent = np.mean(-y * np.log(yhat) - (1 - y) * np.log(1 - yhat))
    num_correct = np.equal((yhat > .5).astype(int), y).sum()
    accuracy = num_correct / X.shape[0] * 100
    return xent, accuracy


# Inputs.
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# XOR
# y = np.array([[0],
#               [1],
#               [1],
#               [0]])

# AND
y = np.array([[0],
              [0],
              [0],
              [1]])

# Shapes and sizes.
N, D = X.shape
K = y.shape[-1]

# Weights and bias from `X` to `yhat`.
W_X_yhat = np.random.normal(scale=0.01, size=[D, K])
b_X_yhat = np.zeros([1, K])

step_size = 1e-1
epochs = int(1e4)
for epoch in range(1, epochs + 1):

    yhat = forward(X, W_X_yhat, b_X_yhat)

    # Evaluate ten times during training.
    if not epoch % (epochs // 10):
        xent, accuracy = eval(y, yhat)
        print('xent / acc: {:.10f} / {:.0f} %'.format(xent, accuracy))

    dW_X_yhat, db_X_yhat = backward(X, y, yhat)

    W_X_yhat += -step_size * dW_X_yhat
    b_X_yhat += -step_size * db_X_yhat

~~~
