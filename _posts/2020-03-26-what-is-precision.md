---
title: What is the precision matrix? 
layout: single
author_profile: false
mathjax: true
---

### Refreshing my understand of the meaning of the [precision](https://en.wikipedia.org/wiki/Precision_(statistics)) matrix.

This became relevant in [my ongoing project](https://tmorville.github.io//rl-svd_1/) with multi-armed contextual bandits.

I want to answer three questions.

1. How does covariance intuitively relate to precision?
2. What does the precision matrix tell us in bayesian linear regression?
3. What does noisy data look like in a precision matrix?

Covariance is defined as the expected value (or mean) of the product of their deviations from their individual expected values

![png]({{ site.url }}{{ site.baseurl }}/assets/images/precision-eq-1.png)

and precision is the matrix inverse of the covariance matrix

![png]({{ site.url }}{{ site.baseurl }}/assets/images/precision-eq-2.png)

```python
import numpy as np

import matplotlib.pyplot as plt

linalg = np.linalg
```

To answer the questions above we need to generate some simulated data. The `data_generator()` function uses [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) to generate a correlated and a uncorrelated matrix we can use to illustrate the difference between covariance and precision.


```python
def data_generator(N, M):
    
    a = np.random.rand(N, M)

    # Positive semi-definite matrix
    cov = np.dot(a.T, a)
    
    L = linalg.cholesky(cov)

    uncorrelated = np.random.standard_normal((N, M)).T
    correlated = np.dot(L, uncorrelated).T
    
    return correlated, uncorrelated
```

To check if this works, plot the data


```python
c, uc = data_generator(100, 100)

fig_scatter, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))

ax1.scatter(c[:, 0], c[:, 1])
ax2.scatter(uc[:, 0], uc[:, 1])
```

![precision-1]({{ site.url }}{{ site.baseurl }}/assets/images/precision-1.png)

Quite clearly, the data generating process works.The LHS figure shows correlated data, and the RHS figure shows a noise pattern.


```python
def make_plot(N, M, text=True):
    
    # Data
    corr, uncorr = data_generator(N, M)
    
    
    for d in [uncorr, corr]:

        # Setup matrices
        m = np.shape(d)[0]
        x = np.matrix(d[:, 1:])
        y = np.matrix(d[:, 0]).T

        lambda_prior = 0.25

        s = np.dot(x.T, x)

        precision_a = s + lambda_prior * np.eye(m-1)
        cov_a = np.linalg.inv(precision_a)
        mu_a = np.dot(cov_a, np.dot(x.T, y))

        fig_matrix, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 15))
                
        ax1.matshow(cov_a)
        ax1.set_title('Covariance matrix')
        
        ax2.matshow(precision_a)
        ax2.set_title('Precision matrix')
        
        cax = fig_matrix.add_axes([ax2.get_position().x1+0.01,ax2.get_position().y0,0.02,ax2.get_position().height])
        fig_matrix.colorbar(ax1.matshow(cov_a), cax=cax)
                
        if text:

            for (i, j), z in np.ndenumerate(cov_a):
                ax1.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

            for (i, j), z in np.ndenumerate(precision_a):
                ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')    
```


```python
make_plot(100, 100, text=False)
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/precision-2.png)



![png]({{ site.url }}{{ site.baseurl }}/assets/images/precision-3.png)


The top two matrices are the uncorrelated data, and the bottom two matrices are the correlated data. Straight off the bat, we notice that the precision matrix shows the largest difference between uncorrelated and correlated data. The lines that permeate the precision matrix in the bottom RHS are clearly correlated (yellow colors) while the blue colors in the top RHS precision matrix are clearly very uncorrelated. 

Looking to the difference between the covariance and precision matrix for the top uncorrelated data, we notice that there are small square "islands" in the covariance matrix that does not exist in the precision matrix. The precision matrix yields **partial correlations** of variables, while the covariance matrix yields **unconditional correlation** between variables. 

Consider this example.

There are three events **A**, **B** and **C**. A being that the grass in your front yard is wet, **B** that your driveway is wet and **C** the fact that it rained. Now, if we just look at **A** and **B**, they will be heavily correlated, but once we condition on **C**, they are pretty much uncorrelated. *A partial correlation describes the correlation between two variables, after you have conditioned on all other variables*. So what we are seeing the in the uncorrelated data are conditionally independent variables, while we are seeing conditionally dependent variables in the correlated data.
