---
title: Inverted pendulum using learning
layout: single
---

### The inverted pendulum problem from [OpenAI Gym](https://gym.openai.com/envs/CartPole-v0) is solved using a simple network that learns the relevant derivatives using only positional data. Using a simple state-based policy, this is sufficient to solve the problem with explicit optimal control (work in progress).

### To-do list: 
Train a fully connected network on (essentially noisy) positional data from CartPole-v0 to learn first and second derivatives of cart position $$\x_t$$ and pole angle $$\theta_t$$.
