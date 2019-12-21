# AI-Cube

## What is AI-Cube

AI-Cube is a project using RL (reinforcement learning) method to solve a 2x2 Rubik's Cube.

## How it works

The project was inspired by the article [Solving the Rubik's Cube Without Human Knowledge](https://arxiv.org/abs/1805.07470), and basically followed the instructions provided by the above article.

However due to the humble computation power I own, I only have a microsoft surface laptop without GPU, I adapted the method from the paper, which was aiming at a 3x3x3 Rubik's Cube, to solving a 2x2x2 Rubik's Cube.

## How to run the demo

Just make sure you have all the required libraries installed, then run demo.py.

The demo.py will scramble the cube 20 times and try to solve it back. I set it to 20 times because it is easy to perform with a real cube on class.

##

## Requirements

* Python 3.5+
* PyTorch 1.3.1+
* NumPy (intergrated within Python)
