---
title: Inverted pendulum using learning
layout: single
---

### The inverted pendulum problem from [OpenAI Gym](https://gym.openai.com/envs/CartPole-v0) is solved using a simple network that learns the relevant derivatives using positional data. The inferred derivatives are used in a RL-scheme which keeps the cart stable. 

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### To-do list: 

- Train a fully connected, one-layer network on (essentially noisy) positional data from CartPole-v0 to learn first and second derivatives of cart position $$x_t$$ and pole angle $$\theta_t$$ (this is given in OpenAI). The output of this network will be the probability of a derivative $$P(\dot{x_t})$$ and $$P(\dot{\theta_t})$$ and not the derivative itself. 
- Use the output of the network _on the fly_ combined with [Q-learning](https://en.wikipedia.org/wiki/Q-learning) to keep the pole stable. 