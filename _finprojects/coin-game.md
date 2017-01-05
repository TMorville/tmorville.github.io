---
title: Non-ergocidity in a simple coin game
layout: single
---

### Based on a [seminal talk](https://www.youtube.com/watch?v=f1vXAHGIpfc) by Ole Peters, I illustrate the fundamental difference between an ergodic and non-ergodic process using a simple coin game.  

---

Game dynamics: 

* At each time step a fair coin is flipped and lands either heads or tails.
* If heads show up, the amount initially gambled will **increase with 50%**. 
* If the coin lands tails, the initial amount will **decrease by 40%**. 

Assuming that initial wealth is 1. the following code plots the outcome of $$N$$ different coin games over $$T$$ time. (If you download this notebook, you can add more time steps by sliding the time-bar right, the function iterates the current wealth according to the dynamic explained above.) 

In the following code, we assume that 

* $$W_t$$ is the wealth process at time $t$ (W_ in code)
* $$\mathbb{E}$$ is the large-N limit expectation average (E_ in code)
* $$\mathcal{A}$$ is the the long-time average (A_ in code)
* $$\mu$$ is the emperical mean (mu in code)


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

sns.set(style="white", palette="muted", color_codes=True)

%matplotlib inline
```


```python
def multiplicativeW( T, N, W, E, A, mu, phi ):
    
    w0 = 1 
    c1 = 0.5 
    c2 = -0.4
    
    j = N
    k = T
    
    size = (N,T)
    
    coinToss = np.zeros(size)
    
    W_ = np.zeros(size)
    E_ = np.zeros(T)
    A_ = np.zeros(T)
    mu_ = np.zeros(T)
    phi_ = np.zeros(size)
    
    np.random.seed(1); 
    
    for j in range(N):
        
        
        
        for k in range(T):
            if k == 0:
                
                W_[j,k] = w0
                E_[k] = w0
                A_[k] = w0
                mu_[k] = w0
                
            else:
                
                # coin dynamics 
                coinToss[j,k] = np.random.choice([c1,c2])
                
                # wealth dynamics
                W_[j,k] = W_[j,k-1] * (1+coinToss[j,k])
                
                # theoretical constants
                E_[k] = w0 + np.power(1+((0.5*c1)+(0.5*c2)),[k])
                A_[k] = np.power(np.exp((np.log(1+c1)+np.log(1+c2))/2),[k])
                
                # emperical mean
                mu_[k] = np.mean(W_[:,k])
    
    fig = plt.figure(figsize=(12,5))
    plt.yscale('log', nonposy='clip')
    
    NUM_COLORS = N
    cm = sns.cubehelix_palette(8, start=.5, rot=-.75,as_cmap = True)
    
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    
    if W is True:
    
        for j in range(N):
            ax1.plot(W_[j],linewidth = 1)
            sns.despine()
        
    if E is True:    
        ax1.plot(E_,linewidth = 2,color='k',label = (r'$\mathbb{E}$'))
        plt.legend(loc='upper left', fontsize=14)
        
    if A is True:
        ax1.plot(A_,linewidth = 2,color='r', label = (r'$\mathcal{A}$'))
        plt.legend(loc='upper left', fontsize=14)
        
    if mu is True:
        ax1.plot(mu_,linewidth = 2,color='r', label = (r'$\mu$'))
        plt.legend(loc='upper left', fontsize=14)
        
```

First, lets assume that just one player plays this coin game once every minute for an hour and plot $$N=1$$ different instances of the above coin-game for $$T=60$$ time periods. 


```python
interact(multiplicativeW, T = 60, N = 1, W = True, E = False, A = False, mu = False, phi = False)
```

![png]({{ site.url }}{{ site.baseurl }}/assets/images/SimpleCoinGame_6_1.png)


Its clear that this process got quite close to very within the hour! However, it seems very noisy and its hard to make out if this one trajectory was just a bit unfortunate. Lets _repeat_ the game 100 times and see what happens.


```python
interact(multiplicativeW, T = 60, N = 100, W = True, E = False, A = False, mu = False, phi = False)
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/SimpleCoinGame_8_0.png)


We're still not getting much smarter. Well, lets try to average the 100 different trajectories and see what that looks like.


```python
interact(multiplicativeW, T = 60, N = 100, W = True, E = False, A = False, mu = False, phi = False)
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/SimpleCoinGame_10_0.png)



```python
interact(multiplicativeW, T = 60, N = 10000, W = True, E = False, A = False, mu = False, phi = False)
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/SimpleCoinGame_11_0.png)



