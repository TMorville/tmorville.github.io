---
title: 
layout: single
---

### An attempt to implement the recurrent attention model (RAM) from "Recurrent Models of Visual Attention" (Mnih+ 2014) in collaboration with [ppries](https://github.com/ppries) (work in progress).

Link to [repo](https://github.com/ppries/tensorflow_mnist_ram) and my [notes](https://tmorville.github.io//assets/deepmind.pdf).

### To-do list: 

- Overarching aim is to implement a slightly more refined reward structure, such that the learner is _punished_ by wrong guess. The biological analogy is that an animal increasing it's survivability by correctlying inferring the _important_ features of the environment. Not doing so will decrease accumulated reward. E.g. water is important when thirsty, but less so when hungry. 
- Adapt [sean999 implementation](https://github.com/seann999/tensorflow_mnist_ram) to python 3.x _(done, see [repo](https://github.com/ppries/tensorflow_mnist_ram))_. 
- Change reward to episodic structure with focus on only two numbers, e.g. 3 and 7. 
- Introduce dynamics into reward such that after e.g. **T** episodes, 3 stops being rewarding (and becomes punishing) and 7 becomes rewarding. 



