# AI-Cube

## What is AI-Cube

AI-Cube is a project using RL (reinforcement learning) method to solve a 2x2 Rubik's Cube.

## How it works

The project was inspired by the article [Solving the Rubik's Cube Without Human Knowledge](https://arxiv.org/abs/1805.07470), and basically followed the instructions provided by the above article.

However due to the humble computation power I own, I only have a microsoft surface laptop without GPU, I adapted the method from the paper, which was aiming at a 3x3x3 Rubik's Cube, to solving a 2x2x2 Rubik's Cube.

## How to run the demo

Just make sure you have all the required libraries installed, then run demo.py.

The demo.py will scramble the cube 20 times and try to solve it back. I set it to 20 times because it is easy to perform with a real cube on class.

## Cube representation

I used a similiar representation as the above paper did, i.e. stickers tracking representation. Refer to the paper if you are interested.

### Why not this representation

I know someone would ask why I did not use the expansion form of a cube.

![](https://github.com/xiahualiu/AI-Cube/blob/master/cube.png?raw=true)

Because it is not efficient, the expansion form space is much more bigger than the real cube permutation space. [This article](https://medium.com/datadriveninvestor/reinforcement-learning-to-solve-rubiks-cube-and-other-complex-problems-106424cf26ff) will show you the reason.

By tracking 7 purple stickers on top and bottom. We can define any cube state.

![](https://github.com/xiahualiu/AI-Cube/blob/master/stickers.png?raw=true)

The left top grey sticker was not tracked because it is used as the reference sticker, it is fixed during scrambling and solving. We fix it, becase we want the solved state unique, and using reference stickers can eliminate the rotation effect. 

If we track all 8 stickers, then the solved cube (actually, any given cube) can be represented by different 24 sticker states, which is caused by the rotation of the cube.

## Net structure

![](https://github.com/xiahualiu/AI-Cube/blob/master/net.png?raw=true)

All layers are linear layers and end with *elu* activation function.

Left 6 outputs are policy output, the output on the right is the current value output.

## Performance

Well, sometimes it failes because I limited the searching tree size in order to increase speed. You can cancel the searching tree cap and get a 100% sucessful rate. I tested the solver ( solving time limited to 1 minute) and here is the performance:

![](https://github.com/xiahualiu/AI-Cube/blob/master/result.png?raw=true)

## TODO

The Monte Carlo Tree Searching section was hurried in one night, I saw people using asychoronous Monte Carlo Tree Searching, it will increase the searching speed much more, but I did not use it. I will add it later. (I am kinda lazy, so no guarantee here)

## Requirements

* Python 3.5+
* PyTorch 1.3.1+
* NumPy (intergrated with Python)

