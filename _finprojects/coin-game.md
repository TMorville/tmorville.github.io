---
title: Non-ergocidity in a simple coin game
layout: single
---

### Based on a [seminal talk](https://www.youtube.com/watch?v=f1vXAHGIpfc) by Ole Peters I illustrate the fundamental difference between an ergodic and non-ergodic process using a simple coin game.  

Game dynamics: 

>*At each time step a fair coin is flipped and lands either heads or tails.
>*If heads show up, the amount initially gambled will **increase with 60%**. 
>*If the coin lands tails, the initial amount will **decrease by 40%**. 

Plots trajectories of exactly this game, assuming that initial wealth is 1. As you add more time steps by sliding the time-bar right, the function iterates the current wealth according to the dynamic explained above. 


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
def multiplicativeW(T,N):
    
    w0 = 1 
    c1 = 0.6 
    c2 = -0.4
    
    j = N
    k = T
    
    size = (N,T)
    
    coinToss = np.zeros(size)
    
    Wmul = np.zeros(size)
    Emul = np.zeros(T)
    expMul = np.zeros(T)
    np.random.seed(1); 
    
    for j in range(N):
        for k in range(T):
            if k == 0:
                
                Wmul[j,k] = w0
                Emul[k] = w0
                expMul[k] = w0
                
            else:
                #CONSTANTS
                Emul[k] = w0 + np.power(1+((0.5*c1)+(0.5*c2)),[k])
                expMul[k] = np.power(np.exp((np.log(1+c1)+np.log(1+c2))/2),[k])

                #GENERATE COIN-TOSS SEQUENCE
                coinToss[j,k] = np.random.choice([c1,c2])

                #MULTIPLICATIVE DYNAMICS 
                Wmul[j,k] = Wmul[j,k-1] * (1+coinToss[j,k])

    fig = plt.figure(figsize=(12,5))
    
    NUM_COLORS = N
    cm = sns.cubehelix_palette(8, start=.5, rot=-.75,as_cmap = True)
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    
    for j in range(N):
        plt.yscale('log', nonposy='clip')
        ax1.plot(Wmul[j],linewidth = 1)
        sns.despine()
        
    ax1.plot(Emul,linewidth = 2,color='k',label = 'Expected Average Growth')
    ax1.plot(expMul,linewidth = 2,color='r', label = 'Realised Average Growth')
    plt.legend(loc='upper left', fontsize=14)
```


```python
interact(multiplicativeW, T = 500, N = 50)
```


![png]({{ site.url }}{{ site.baseurl }}/assets/images/SimpleCoinGame_4_0.png)



```python

```
