---
title: Inverted pendulum using learning
layout: default
---

Does this text appear as? 

~~~ python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

import gym
env = gym.make('CartPole-v0')

for i_episode in range(10):
    x_t, grad_x_t, theta_t, grad_theta_t = env.reset()
    for t in range(250):
        env.render()
        action = env.action_space.sample()
        
        decider = grad_x_t
        if np.abs(grad_theta_t) > np.abs(grad_x_t):
            decider = grad_theta_t
        
        action = 1 if decider > 0 else 0
        (x_t, grad_x_t, theta_t, grad_theta_t), reward, done, info = env.step(action)
        #print(x_t, grad_x_t, theta_t, grad_theta_t)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
            

~~~

