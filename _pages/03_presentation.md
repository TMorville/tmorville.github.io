---
layout: page
title: Presentation
permalink: /Presentation/
---

#This is a testslide with bullets
** test


```python
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline


x=np.linspace(0,1,100)
f=2

def pltsin(f):
    plt.plot(x,np.sin(2*np.pi*x*f))
```


```python
interact(pltsin, f=(1,10,0.1));
```


![png](presentation_files/presentation_2_0.png)



```python

```
