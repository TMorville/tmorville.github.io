---
title: Learning using gradient descent on a free-energy potential
layout: single
---

### Based on [K. Friston](http://www.fil.ion.ucl.ac.uk/~karl/A%20free%20energy%20principle%20for%20the%20brain.pdf) and [R. Bogacz](http://www.sciencedirect.com/science/article/pii/S0022249615000759) a learning scheme is implemented using gradient descent on a free-energy potential. In the following the framework is guided by biological semantics to maximise intuition - but generally this method can (and has been) applied to many different problems.  

If you are familiar with Bayes, but not approximate inference, I suggest that you skip **Part I** and head to **Part II**. If you're familiar with both, you probably won't learn much in the following. Lets begin!

### Part I - Bayes
Consider a theoretical one-dimensional thermoregulator. This simple organism maximises survival by maximising sojourn time in some optimal temperature state defined on evolutionary time. It does this by simple error based control, like a thermostat on a heater. Thus the only signal the regulator cares about, is the real (euclidian) distance between its current temperature state and the optimal state. This real distance on $$\mathbb{R}$$ is the homeostatic error $$\epsilon$$ and this is communicated via. a noisy efferent signal $$s$$. The non-linear function $$g(\epsilon)$$ relates homeostatic error to percieved efferent signal, such that when homeostatic error is $$\epsilon$$ the percieved efferent signal is normally distributed with mean $$g(\epsilon)$$ and variance $$\Sigma_\epsilon$$. Thus, the likelihood function is 

$$ p(s|\epsilon)=f(s;g(\epsilon),\Sigma_{s})$$

where

$$ f(x;\mu,\Sigma)=\frac{1}{\sqrt{2\pi\Sigma}}\mbox{exp}\left(-\frac{(x-\mu)^{2}}{2\Sigma}\right). $$
 
Through evolutionary filtering, the agent has been endowed with strong priors on its interoceptive states and therefore expects homeostatic error to normally distributed with mean $$\epsilon_{p}$$ and $$\Sigma_{p}$$ where the subscript $$p$$ stands for prior. Formally $$p(\epsilon)=f(\epsilon;\epsilon_{p},\sigma_{p})$$.
 
To compute the exact distribution of sensory input $$s$$ we can formulate the posterior using Bayes theorem 

$$p(\epsilon|s)=\frac{p(\epsilon)p(s|\epsilon)}{p(s)}$$
 
where the denominator is 

$$p(s)=\int p(\epsilon)p(s|\epsilon)d\epsilon$$
 
and sum the whole range of possible $$\epsilon$$.

The following code implements such an exact solution and plots it.

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

sns.set(style="white", palette="muted", color_codes=True)

%matplotlib inline
```


```python
# non-linear transformation of homeostatic error to percieved sensory input e.g. g(phi)
def sensory_transform(input):
        
    sensory_output = np.square(input)
        
    return sensory_output
```


```python
def exact_bayes():
    
    # variabels 
    epsilon = 2 # observed homeostatic error 
    sigma_e = 1 # standard deviation of the homeostatic error
    epsilon_p = 3 # mean of prior homeostatic error / noisy input / simple prior
    sigma_s = 1 # variance of prior / sensory noise 
    s_range = np.arange(0.01,5,0.01) # range of sensory input
    s_step = 0.01 # step size
        
    # exact bayes (equation 4)
    numerator = (np.multiply(norm.pdf(s_range,epsilon_p,sigma_s),# prior
                            norm.pdf(epsilon,sensory_transform(s_range),sigma_e))) # likelihood
    normalisation = np.sum(numerator*s_step) # denominator / model evidence / p(noisy input) (equation 5)
    posterior = numerator / normalisation # posterior
    
    # plot exact bayes
    plt.figure(figsize=(7.5,2.5))
    plt.plot(s_range,posterior)
    plt.ylabel(r' $p(\epsilon | s)$')
    sns.despine()
```


```python
exact_bayes()
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/free_energy_homeostasis_3_0.png)



```python
def simple_dyn():
    
    # variabels 
    epsilon = 2 # observed homeostatic error 
    sigma_e = 1 # standard deviation of the homeostatic error
    epsilon_p = 3 # mean of prior homeostatic error / noisy input / simple prior
    sigma_s = 1 # variance of prior / sensory noise 
    s_range = np.arange(0.01,5,0.01) # range of sensory input
    s_step = 0.01 # step size
    
    # assume that phi maximises the posterior 
    phi = np.zeros(np.size(s_range))
    
    # use Eulers method to find the most likely value of phi
    for i in range(1,len(s_range)):
        
        phi[0] = epsilon_p
        phi[i] = phi[i - 1] + s_step * ( ( (epsilon_p - phi[i - 1]) / sigma_e ) +
        ( ( epsilon - sensory_transform(phi[i - 1]) ) / sigma_e ) * (2 * phi[i - 1]) ) # equation 12
    
    # plot convergence
    plt.figure(figsize=(5,2.5))
    plt.plot(s_range,phi)
    plt.xlabel('Time')
    plt.ylabel(r' $\phi$')
    sns.despine()
    
```


```python
simple_dyn()
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/free_energy_homeostasis_5_0.png)



```python
def learn_phi():
    
    # variabels 
    epsilon = 2 # observed homeostatic error 
    sigma_e = 1 # standard deviation of the homeostatic error
    epsilon_p = 3 # mean of prior homeostatic error / noisy input / simple prior
    sigma_s = 1 # variance of prior / sensory noise 
    s_range = np.arange(0.01,5,0.01) # range of sensory input
    s_step = 0.01 # step size
    
    # preallocate
    phi = np.zeros(np.size(s_range)) 
    xi_e = np.zeros(np.size(s_range)) 
    xi_s = np.zeros(np.size(s_range))
    
    # dynamics of prediction errors for homeostatic error (xi_e) and sensory input (xi_s)
    for i in range(1,len(s_range)):
        
        phi[0] = epsilon_p # initialise best guess (prior) of homeostatic error
        xi_e[0] = 0 # initialise prediction error for homeostatic error
        xi_s[0] = 0 # initialise prediction error for sensory input
        
        phi[i] = phi[i-1] + s_step*( -xi_e[i-1] + xi_s[i-1] * ( 2*(phi[i-1]) ) ) # equation 12
        xi_e[i] = xi_e[i-1] + s_step*( phi[i-1] - epsilon_p - sigma_e * xi_e[i-1] ) # equation 13
        xi_s[i] = xi_s[i-1] + s_step*( epsilon - sensory_transform(phi[i-1]) - sigma_s * xi_s[i-1] ) # equation 14
    
    # plot network dynamics
    plt.figure(figsize=(5,2.5))
    plt.plot(s_range,phi)
    plt.plot(s_range,xi_e)
    plt.plot(s_range,xi_s)
    plt.ylabel('Activity')
    sns.despine()                
```


```python
learn_phi()
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/free_energy_homeostasis_7_0.png)



```python
def learn_sigma():
    
    # variabels 
    epsilon = 2 # observed homeostatic error 
    sigma_e = 1 # standard deviation of the homeostatic error
    epsilon_p = 3 # mean of prior homeostatic error / noisy input / simple prior
    sigma_s = 1 # variance of prior / sensory noise 
    s_range = np.arange(0.01,5,0.01) # range of sensory input
    s_step = 0.01 # step size
    
    # new variabels 
    maxt = 20 # maximum number of iterations
    trials = 2000 # of trials 
    epi_length = 20 # length of episode
    alpha = 0.01 # learning rate
    
    mean_phi = 5 # the average value that maximises the posterior
    sigma_phi = 2 # the variance of phi
    last_phi = 5 # the last observed phi
    
    # preallocate
    sigma = np.zeros(trials)
    error = np.zeros(trials)
    e = np.zeros(trials)
    
    sigma[0] = 1 # initialise sigma in 1 
    
    for j in range(1,trials):
        
        error[0] = 0 # initialise error in zero
        e[0] = 0 # initialise interneuron e in zero
        phi = np.random.normal(5, np.sqrt(2), 1) # draw a new phi every round 
        
        for i in range(1,2000):
            
            error[i] = error[i-1] + s_step*(phi-last_phi-e[i-1]) # equation 59 in Bogacz
            e[i] = e[i-1] + s_step*(sigma[j-1]*error[i-1]-e[i-1]) # equation 60 in Bogacz
            
        sigma[j] = sigma[j-1] + alpha*(error[-1]*e[-1]-1) # synaptic weight (Sigma) update
        
    # plot dynamics of Sigma
    plt.figure(figsize=(5,2.5))
    plt.plot(sigma)
    plt.xlabel('Time')
    plt.ylabel(r' $\Sigma$')
    sns.despine()           


```


```python
learn_sigma()
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/free_energy_homeostasis_9_0.png)





