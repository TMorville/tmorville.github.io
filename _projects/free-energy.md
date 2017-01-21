---
title: Learning using gradient descent on a free-energy potential
layout: single
---

### Based on [K. Friston](http://www.fil.ion.ucl.ac.uk/~karl/A%20free%20energy%20principle%20for%20the%20brain.pdf) and following [R. Bogacz](http://www.sciencedirect.com/science/article/pii/S0022249615000759) a learning scheme is implemented using gradient descent on a "free-energy" potential.  

---

### Why is this important/exciting?

[R. Bogacz](http://www.sciencedirect.com/science/article/pii/S0022249615000759) delivers the most explicit and beautifully made tutorial on a subject that can be very difficult to understand - [Variational Bayes](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) - as seen from Karl Fristons point of view, using the [Free-Energy principle](https://en.wikipedia.org/wiki/Free_energy_principle) to motivate Active Inference, which has come one of the dominating hypothesis of how the brain works. 

In the following I solve the excersises given in [R. Bogacz](http://www.sciencedirect.com/science/article/pii/S0022249615000759) and expand them further to motivate a broarder view on Active Learning in ML, and how this relates to Active Inference. 

### Nomenclature:

I use the same notation as in R. Bogacz and the code refers to equations in the paper. 

---

### Part I - Bayes
The likelihood function (probability of a signal given a homeostatic error) is defined as 

$$ p(s|\epsilon)=f(s;g(\epsilon),\Sigma_{s})$$

where

$$ f(x;\mu,\Sigma)=\frac{1}{\sqrt{2\pi\Sigma}}\mbox{exp}\left(-\frac{(x-\mu)^{2}}{2\Sigma}\right). $$
 
Through evolutionary filtering, the agent has been endowed with strong priors on its interoceptive states and therefore expects homeostatic error to normally distributed with mean $$\epsilon_{p}$$ and $$\Sigma_{p}$$ where the subscript $$p$$ stands for prior. Formally $$p(\epsilon)=f(\epsilon;\epsilon_{p},\sigma_{p})$$.
 
To compute the exact distribution of sensory input $$s$$ we can formulate the posterior using Bayes theorem 

$$p(\epsilon|s)=\frac{p(\epsilon)p(s|\epsilon)}{p(s)}$$
 
where the denominator is 

$$p(s)=\int p(\epsilon)p(s|\epsilon)d\epsilon$$
 
and sum the whole range of possible $$\epsilon$$.

The following code implements such an exact solution and plots it. Firstly we import some dependencies:

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

sns.set(style="white", palette="muted", color_codes=True)

%matplotlib inline
```
and then define $$g(\cdot)$$:

```python
# non-linear transformation of homeostatic error to percieved sensory input e.g. g(phi)
def sensory_transform(input):
        
    sensory_output = np.square(input)
        
    return sensory_output
```
The reason we explicitly define $$g(\cdot)$$ is that we might want to change it later. For now we assume a simple non-linear relation $$g(\epsilon)=\epsilon^2$$. The following snippet of code assumes values of $$\epsilon,\Sigma_e,\epsilon_p,\Sigma_s$$ and plots tbe posterior distribtuion $$p(\epsilon|s)$$.

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

Inspecting the graph, we find that approximately $$\phi=1.6$$ maximises the posterior. There are two fundamental problems with this approach 

 1. The posterior does not take a standard form, and is thus described by (potentially) infinitely many moments, instead of just simple sufficient statistics, such as the mean and the variance of a gaussian.

 2. The normalisation term that sits in the numerator of Bayes formula 
 
$$p(s)=\int p(\epsilon)p(s|\epsilon)d\epsilon$$

can be complicated and numerical solutions often rely on computationally intense algorithms, such as the [Expectation-Maximisation algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm). 

---

### Part II - Approximate inference
We are interested in a more general way of finding the value that maximises the posterior $$\phi$$. This involves maximising the numerator of Bayes equation. As this is independent of the denominator and therefore maximising $$p(\epsilon)p(s|\epsilon)$$ will maximise the posterior. By taking the logarithm to the numerator we get 

$$
F=\mbox{ln}p(\phi)+\mbox{ln}p(s|\phi)
$$

and the dynamics can be derived (see notes) to be  

$$
\dot{\phi}=\frac{\epsilon_{p}-\phi}{\Sigma_{p}}+\frac{s-h(\phi)}{\Sigma_{s}}g^{'}(\phi)
$$

The next snippit of code asumes values for $$\epsilon,\Sigma_e,\epsilon_p,\Sigma_s$$ and implements the above dynamics to find the value of $$\phi$$ that maximises the posterior using a manual implementation of the dynamics and iterating using Eulers method. 


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

It is clear that the output converges rapidly to $$\phi=1.6$$, the value that maximises the posterior. 

So we ask the question: What does a minimal and _biologically plausible_ network model that can do such calculations look like? 

---

### Part III - Learning $$\phi$$ with a network model
Firstly, we must specify what exactly biologically plausible means. 1) A neuron only performs computations on the input it is given, weighted by its synaptic weights. 2) Synaptic plasticity of one neuron is only based on the activity of pre-synaptic and post-synaptic activity connecting to that neuron. 

Consider the dynamics of a simple network that relies on just two neurons and is coherent with the above requirements of local computation


$$ \dot{\xi_{p}} = \phi-\epsilon_{p}-\Sigma_{p}\xi_{p} $$

$$ \dot{\xi_{s}} = s-h(\phi)-\Sigma_{s}\xi_{s} $$

where $$\xi_{p}$$ and $$\xi_{s}$$ are the _prediction errors_

$$ \xi_{p} = \frac{\epsilon_{p}-\phi}{\Sigma_{p}} $$

$$ \xi_{s} = \frac{s-g(\phi)}{\Sigma_{s}} $$

that arise from the assumption that the input is normally distributed (again, see notes for derivations). The next snippit of code implements those dynamics and thus, the network "learns" what value of $$\phi$$ that maximises the posterior. 


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

As the figure shows, the network learns <span style="color: blue">$$\phi$$</span> but is slower in converging than when using Eulers method, as the model relies on several nodes that are inhibits and excites each other which causes oscillatory behaviour. Both <span style="color:green">$$\xi_p$$</span> and <span style="color:red">$$\xi_\epsilon$$</span> oscillate and converges to the values where

$$ \phi-\epsilon_{p}-\Sigma_{p}\xi_{p} = 0 $$

$$ s-h(\phi)-\Sigma_{s}\xi_{s} = 0 $$

---
    
### Part IV - Learning $$\Sigma$$ with a network model

Recall that we assumed that homeostatic error $$\epsilon$$ was communicated via. a noisy efferent signal $$s$$ that we assumed to be normally distributed. Above, we outlined a simple sample method for finding the mean value $$\phi$$ that maximises the posterior $$p(\epsilon\vert s)$$. 

By expanding this simple model, we can esimate the variance $$\Sigma$$ of the normal distribution as well. Considering computation in one single node computing prediction error 

$$ \xi_{i}=\frac{\phi_{i}-g(\phi_{i+1})}{\Sigma_{i}} $$
 

where $$\Sigma_{i}=\left\langle (\phi_{i}-g_{i}(\phi_{i+1})^{2}\right\rangle$$ is the variance of homeostatic error $$\phi_{i}$$. Estimation of $$\Sigma$$ can be achieved by adding a interneuron $$e_{i}$$ which is connected to the prediction error node, and receives input from this via the connection with weight encoding $$\Sigma_{i}$$. The dynamics are described by

$$ \dot{\xi_{i}} = \phi_{i}-g(\phi_{i+1})-e_{i} $$

$$ \dot{e} = \Sigma_{i}\xi_{i}-e_{i} $$

which the following snippit of code implements. 
 

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

Because $$\phi$$ is constantly varying <code>phi = np.random.normal(5, np.sqrt(2), 1)</code> $$\Sigma$$ never does not converge to just one value, but instead to _approximately_ 2, the variance of $$\phi_i$$.



