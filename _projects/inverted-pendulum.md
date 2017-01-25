---
title: Inverted pendulum using learning
layout: single
---

### The inverted pendulum problem from [OpenAI Gym](https://gym.openai.com/envs/CartPole-v0) is solved using a simple network that learns the relevant derivatives using positional data. The inferred derivatives are used in a RL-scheme which keeps the cart stable. 

---

### Why is this important/exciting?

Optimal control theory is having a renascence in complex network analysis. There are many very good reasons for this, and if you're interested, I suggest that you read [this excellent article](https://arxiv.org/abs/1508.05384) by Yang-Yu Liu and Albert-Laszló Barabási. 

This project is meant to increase my own (and perhaps your own) understanding of a simple problem: The inverted pendulum or cart-pole problem. This can be solved explicitly using optimal control theory. To make things a bit more difficult for our selves, and hopefully discover or learn something in the process, we "overlay" a neural-network over the variabels of interest, such that the network must learn the relevant control parameters. 

---

### To-do list: 

- Train a fully connected, one-layer network on (essentially noisy) positional data from CartPole-v0 to learn first and second derivatives of cart position $$x_t$$ and pole angle $$\theta_t$$ (this is given in OpenAI). The output of this network will be the probability of a derivative $$P(\dot{x_t})$$ and $$P(\dot{\theta_t})$$ and not the derivative itself. 
- Use the output of the network _on the fly_ combined with [Q-learning](https://en.wikipedia.org/wiki/Q-learning) to keep the pole stable. 