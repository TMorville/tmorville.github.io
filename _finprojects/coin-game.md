---
title: Non-ergocidity in a simple coin game
layout: single
---

### Based on a [seminal talk](https://www.youtube.com/watch?v=f1vXAHGIpfc) by Ole Peters, I illustrate the fundamental difference between an ergodic and non-ergodic process using a simple coin game.  

(_If you download this notebook, you can use the interactive widgets_) 

---

### Dynamics: 

* At each time step a **fair coin** is flipped and lands either heads or tails.
* For **heads** wealth **increases with 50%** e.g. $$W_t\cdot1.5$$
* For **tails** wealth **decreases by 40%** e.g. $$W_t\cdot0.6$$

### Nonclamenture:

* $$W_t$$ is the wealth at time $$t$$ (<code>W_</code>)
* $$\mathbb{E}$$ is the large-N limit expectation average (<code>E_</code>)
* $$\mathcal{A}$$ is the the long-time average (<code>A_</code>)
* $$\mu$$ is the emperical mean (<code>mu</code>)

---

### Initial analytics:


---

### Simulations:

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

First, let's assume that we play the coin game once every minute for an hour and plot just one ($$N=1$$) instance of the dynamic outlined above for $$T=60$$ time periods. 


```python
interact(multiplicativeW, T = 60, N = 1, W = True, E = False, A = False, mu = False, phi = False)
```

![png]({{ site.url }}{{ site.baseurl }}/assets/images/SimpleCoinGame_6_1.png)


Its clear that this process got close to zero $$(10^{-2})$$ quite quickly (after 50 flips)! However, it's also very noisy and difficult to make out if this one trajectory was just a bit unfortunate. Lets _repeat_ the game 100 times $$(N=100)$$ and see what happens.


```python
interact(multiplicativeW, T = 60, N = 100, W = True, E = False, A = False, mu = False, phi = False)
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/SimpleCoinGame_8_0.png)


We're still not getting much smarter. Looks like half on the trajectories are increasing, while the other half is dereasing, but its quite difficult to tell. Lets try to average the 100 different trajectories and see what that looks like.


```python
interact(multiplicativeW, T = 60, N = 100, W = True, E = False, A = False, mu = False, phi = False)
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/SimpleCoinGame_10_0.png)

Well, that looks very flat at around 0. Lets increase the number of times we play to 10000 instead and see if we can get a more conclusive graphical result. 



```python
interact(multiplicativeW, T = 60, N = 10000, W = True, E = False, A = False, mu = False, phi = False)
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/SimpleCoinGame_11_0.png)

Alright! It seems like the empirical average <span style="color:red">$$\mu$$</span> steadily increases. If you feel like removing more noise, you can increase $$N$$ (however this script is quite slow due to all the if-loops). 

Another thing we could do was to calculate the theoretical mean or _expectation operator_. We will denote this as

$$
\left\langle W(t)\right\rangle _{N} = \frac{1}{N}\sum_{i=1}^{N}W_i(t).
$$



