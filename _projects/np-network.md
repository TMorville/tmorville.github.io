---
title: Manual implementation of a network
layout: single
---

### Implementation of a one-layer neural network that solves the [XOR](https://en.wikipedia.org/wiki/Exclusive_or) task using only numpy, roughly following [this](https://cs231n.github.io/optimization-2/) excellent blogpost by Andrej Karpathy. 

---


~~~python

import numpy as np

input_ = np.asarray([[0,1],[1,0],[0,0],[1,1]])
weights = np.asarray([0.5,0.5])
targets = np.asarray([1,1,0,0])

print(input_.shape)
print(weights.shape)

y_ = np.dot(input_,weights)
print(y_)

def sigmoid():
	out_sig = 1/(1+np.exp(-y_))
	print(out_sig)

	return out_sig

out_sig = sigmoid()

def lossfunc():

	out_loss = -targets*np.log(out_sig)-(1-targets)*(np.log(1-out_sig))
	mean_loss = np.mean(out_loss)
	print(out_loss,mean_loss)

	return out_loss

lossfunc()

~~~
