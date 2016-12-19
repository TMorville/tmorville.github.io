---
title: Learning using gradient descent on a free-energy potential
layout: single
---

(Work in progress)

Based on R. Bogacz and  K. Friston, S. Samothrakis, and R. Montague a simple learning scheme is implemented in python using (error based) gradient descent on a free-energy potential. This is analog to minimising the KL-divergence. It is shown that 1) This simplifies the calculations 2) Learns the parameters of a few simple distributions (dirac and gaussian). 
 


```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
%matplotlib inline
```


```python
sns.set(style="white", palette="muted", color_codes=True)
```


```python
def ex():
    
    mu = 2 # size of input 
    sigma_mu = 1 # variance of input size
    v_p = 3 # expected mean size of input 
    sigma_v = 1 # variance of noisy input signal
    v_range = np.arange(0.01,5,0.01) # range 
    v_step = 0.01 # step size
    
    
    # bayes 
    numerator = np.multiply(norm.pdf(v_range,v_p,sigma_v),norm.pdf(mu,np.square(v_range),sigma_mu)) #prior * likelih
    normalisation = np.sum(numerator*v_step) # model evidence / p(noisy input)
    p = numerator / normalisation # posterior
    
    fig = plt.figure(figsize=(7.5,2.5))
    
    ax = fig.add_subplot(1,3,1)
    plt.plot(v_range,p)
    sns.despine()

    # maximises the posterior 
    phi = np.zeros(np.size(v_range))
    
    for i in range(1,len(v_range)):
        
        phi[0] = v_p
        phi[i] = phi[i - 1] + v_step * ( ((v_p - phi[i - 1]) / sigma_v) +
        ( (mu - np.square(phi[i - 1])) / sigma_mu) * (2 * phi[i - 1]))

    ax = fig.add_subplot(1,3,2)
    plt.plot(v_range,phi)
    sns.despine()
    
    phi = np.zeros(np.size(v_range))
    eps_v = np.zeros(np.size(v_range))
    eps_mu = np.zeros(np.size(v_range))
    
    for i in range(1,len(v_range)):
        
        phi[0] = v_p
        eps_v[0] = 0
        eps_mu[0] = 0
        
        phi[i] = phi[i-1] + v_step*(eps_mu[i-1]*np.square(phi[i-1])-eps_mu[i-1])
        eps_v[i] = eps_v[i-1] + v_step*(phi[i-1]-v_p-sigma_v*eps_v[i-1])
        eps_mu[i] = eps_mu[i-1] + v_step*(mu-np.square(phi[i-1])-sigma_mu*eps_mu[i-1])
    
    ax = fig.add_subplot(1,3,3)
    plt.plot(v_range,phi)
    plt.plot(v_range,eps_v)
    plt.plot(v_range,eps_mu)
    sns.despine()
                
```


```python
ex()
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/Bogacz_3_0.png)



```python
def ex5():
    
    v_step = 0.01
    maxt = 20
    trials = 2000
    
    mean_phi = 5
    sigma_phi = 2
    last_phi = 5
    
    epi_length = 20
    t = 2000
    alpha = 0.02
    
    sigma = np.zeros(trials)
    error = np.zeros(trials)
    e = np.zeros(trials)
    
    sigma[0] = 1
    
    for j in range(1,t):
        
        error[0] = 0
        e[0] = 0
        phi = np.random.normal(5, np.sqrt(2), 1)
        
        for i in range(1,trials):
            
            error[i] = error[i-1] + v_step*(phi-last_phi-e[i-1])
            e[i] = e[i-1] + v_step*(sigma[j-1]*error[i-1]-e[i-1])
            
        sigma[j] = sigma[j-1] + alpha*(error[-1]*e[-1]-1)
        
    plt.plot(sigma)
            


```


```python
ex5()
```

![png]({{ site.url }}{{ site.baseurl }}/assets/images/Bogacz_5_0.png)



