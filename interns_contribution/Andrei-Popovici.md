# Andrei-Sorin Popovici - First Month at RIST

Here I will document what I learned during my first month at the Romanian Institute of Science and Technology

## First week

### Introduction to Recurrent Neural Networks

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

I started with Recurrent Neural Networks by reading one of Andrej Karpathy's blog posts which showcased the effectiveness and versatility of RNNs. He showed that a simple network of standard RNN cells can generate any type of text: Shakespeare, abstract algebra, C++ code and, probably, anything.

![](http://karpathy.github.io/assets/rnn/latex3.jpeg)

### Why are non-linear functions used in Machine Learning

http://cs231n.github.io/neural-networks-case-study/

From one of Karpathy's source codes I discovered this tutorial which explains the cross entropy loss function, it's gradient and why non-linearity is necessary. I also learned how to create a spiral dataset, which might prove useful sometime in the future.

| ![grsgrs](http://cs231n.github.io/assets/eg/spiral_linear.png) | ![enter image description here](http://cs231n.github.io/assets/eg/spiral_net.png "TEST") |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                  Linear (**49%** accuracy)                   |                Non-linear (**98%** accuracy)                 |

### Back again to RNN

http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/

Going back to RNNs I learned that it takes as input **the current information from the dataset and it's previous output state**. This is a concept nicely described by the following formula:

![img](https://latex.codecogs.com/gif.latex?s_t%20%3D%20f%28Ux_%7Bt%7D%20+%20Ws_%7Bt-1%7D%29)

Where ![img](https://latex.codecogs.com/gif.latex?x_t) is the input at the actual time step ![img](https://latex.codecogs.com/gif.latex?t), ![img](https://latex.codecogs.com/gif.latex?s_t) will be the current state, ![img](https://latex.codecogs.com/gif.latex?s_%7Bt-1%7D) was the previous state, ![img](https://latex.codecogs.com/gif.latex?f) is a function used to introduce non-linearity (sigmoid, tanh, ReLU, etc), ![img](https://latex.codecogs.com/gif.latex?U) and ![img](https://latex.codecogs.com/gif.latex?W) are the weight matrices.

http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/

While reading this article I discovered the main disadvantage of this concept: **the vanishing/exploding gradient** on which I will go more in-depth in later (maybe in my second month?).

The next thing natural thing to study was how does and RNN cell looks like, how it works and why it works. 

## Second week

### Implementing a network with LSTM

I implemented an **LSTM** network in Keras which I trained on a toy dataset from the Institute. The results were pretty good, the network being able to create **HTML code with correct syntax**, i.e. it knew that some tags must be closed (`<div>something<\div>`)  and that others should not be closed (`<ul>`). I also used it to generate speech similar to a group of my friends.

http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/

### Studying the LSTM cell

This cell is recurrent in the sense that it not only takes the current input, but also it's previous state. What makes it different and more powerful is it's internal structure: every input gets through some 'gates': input gate, forget gate and output gate.

![Recurrent neural network LSTM tutorial - LSTM cell diagram](http://adventuresinmachinelearning.com/wp-content/uploads/2017/09/LSTM-diagram.png)

https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714

### Latent spaces

At the Institute I heard the term latent space a lot and so I decided to understand what it is. It is a great concept, from both a mathematical and a usability point of view. 

![](https://cdn-images-1.medium.com/max/1000/1*vEZE5VcjUr5RUbt_OWfR_w.gif)

It turns out that the bottleneck of an auto-encoder will learn to place similar objects close and very different objects far away in something similar to vector space.

https://hackernoon.com/latent-space-visualization-deep-learning-bits-2-bd09a46920df

## Third week

### Multi-class multi-label classification

During this week my focus was to implement a CNN which does multi-label multi-class classification. I had to do it on a toy dataset made by Catalin which consisted of ![img](https://latex.codecogs.com/gif.latex?28%20%5Ctimes%2028) images, each with a shape (square, circle or triangle) and each shape had a color (redish, blueish, or greenish).

At first I had a class for each possible combination (![img](https://latex.codecogs.com/gif.latex?3%20%5Ccdot%203%20%3D%209) classes). After that I tried to have a CNN which has two distinct layer as outputs: one for each of the ![img](https://latex.codecogs.com/gif.latex?3) colors, and one for each of the ![img](https://latex.codecogs.com/gif.latex?3) shapes.

![enter image description here](https://lh3.googleusercontent.com/QLsXv1Odlmf79GmISypWX7eASvvvAIWFh4WE1x_uODkXeyxDb_ICxgecG15RhMtHOLUMgxcMG-Fr "CNN with Multi-label classification")

> It turned out that this method is **less computationally expensive** and **as accurate as the first method**

## Fourth week

During this week I tried to implement a symmetric autoencoder with 3 layers (![img](https://latex.codecogs.com/gif.latex?256), ![img](https://latex.codecogs.com/gif.latex?128) and ![img](https://latex.codecogs.com/gif.latex?64) neurons) on each side and a bottleneck layer in the middle. I also tried to experiment with the number of neurons in the middle and then test how the performance improves. At first I used the loss as my main metric, but Cristi pointed out to me that the loss in not necessary relevant and that I should be using something like *'how similar is the decoded image to the label it represents?'*

So, then I trained a CNN to classify the original images with an accuracy of ![img](https://latex.codecogs.com/gif.latex?94%5C%25) and then I tested my autoencoder for each configuration with accuracy as a metric, obtaining the following results

![dgr](https://lh3.googleusercontent.com/MYFhHI4TZdkZnftwPzcE5SaagI5m-_TAEErMXZ2O-HWRThBkFXP7dlLAURnE3yebwFvM5uEaXBBg "Accuracy of the autoencoder compared to the number of neurons is the bottleneck")

### The 2-satisfiabilty problem

The **Satisfiabilty problem (SAT)** is asking if there exists a tuple of boolean values for which a boolean formula is satisfied (it evaluates as *true*). For instance, the following formula:

![](https://latex.codecogs.com/gif.latex?%28%5Cneg%7Bx_%7B1%7D%7D%20%5Cvee%20%5Cneg%7Bx_%7B2%7D%7D%29%20%5Cwedge%20%28x_%7B2%7D%20%5Cvee%20%5Cneg%7Bx_%7B3%7D%7D%29%20%5Cwedge%20%28x_%7B2%7D%20%5Cvee%20x_%7B3%7D%29%20%5Cwedge%20%28%5Cneg%7Bx_%7B2%7D%7D%20%5Cvee%20%5Cneg%7Bx_%7B4%7D%7D%29%20%5Cwedge%20%28x_%7B3%7D%20%5Cvee%20x_%7B4%7D%29)

Is true if we choose the following interpretation for ![img](https://latex.codecogs.com/gif.latex?x_1%2C%20x_2%2C%20x_3) and ![img](https://latex.codecogs.com/gif.latex?x_4):

![img](https://latex.codecogs.com/gif.latex?%5Clangle%20x_%7B1%7D%20%3D%200%2C%20x_%7B2%7D%20%3D%201%2C%20x_%7B3%7D%20%3D%201%2C%20x_%7B4%7D%20%3D%200%20%5Crangle)

The **2-SAT** is a reduction of the above problem to only those formulas which can be brought to a **Conjuctive Normal Form** in which every [clause](https://en.wikipedia.org/wiki/Clause_%28logic%29) has exactly 2 literals.

> The above formula has 4 variables, 5 clauses and 10 literals

We see in the following table that ![img](https://latex.codecogs.com/gif.latex?x_1%20%5Cvee%20x_2) is equivalent to ![img](https://latex.codecogs.com/gif.latex?%5Cneg%20x_1%5Crightarrow%20x_2) and ![img](https://latex.codecogs.com/gif.latex?%5Cneg%20x_2%5Crightarrow%20x_1). We will use this fact in the optimal ![img](https://latex.codecogs.com/gif.latex?O%28N+M%29) solution.

| ![img](https://latex.codecogs.com/gif.latex?x_1) | ![img](https://latex.codecogs.com/gif.latex?x_2) | ![img](https://latex.codecogs.com/gif.latex?x_1%20%5Cvee%20x_2) | ![img](https://latex.codecogs.com/gif.latex?%5Cneg%20x_1%5Crightarrow%20x_2) | ![img](https://latex.codecogs.com/gif.latex?%5Cneg%20x_2%5Crightarrow%20x_1) |
| :----------------------------------------------: | :----------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  ![img](https://latex.codecogs.com/gif.latex?0)  |  ![img](https://latex.codecogs.com/gif.latex?0)  |        ![img](https://latex.codecogs.com/gif.latex?0)        |        ![img](https://latex.codecogs.com/gif.latex?0)        |        ![img](https://latex.codecogs.com/gif.latex?0)        |
|  ![img](https://latex.codecogs.com/gif.latex?0)  |  ![img](https://latex.codecogs.com/gif.latex?1)  |        ![img](https://latex.codecogs.com/gif.latex?1)        |        ![img](https://latex.codecogs.com/gif.latex?1)        |        ![img](https://latex.codecogs.com/gif.latex?1)        |
|  ![img](https://latex.codecogs.com/gif.latex?1)  |  ![img](https://latex.codecogs.com/gif.latex?0)  |        ![img](https://latex.codecogs.com/gif.latex?1)        |        ![img](https://latex.codecogs.com/gif.latex?1)        |        ![img](https://latex.codecogs.com/gif.latex?1)        |
|  ![img](https://latex.codecogs.com/gif.latex?1)  |  ![img](https://latex.codecogs.com/gif.latex?1)  |        ![img](https://latex.codecogs.com/gif.latex?1)        |        ![img](https://latex.codecogs.com/gif.latex?1)        |        ![img](https://latex.codecogs.com/gif.latex?1)        |

We create a *Directed Graph* which has as edges the variables (![img](https://latex.codecogs.com/gif.latex?x_1%2C%20x_2%2C%20x_3%2C%20...)) and their negations (![img](https://latex.codecogs.com/gif.latex?%5Cneg%20x_1%2C%20%5Cneg%20x_2%2C%20%5Cneg%20x_3%2C%20...)). The idea is that for each clause in a formula we will create two directed edges in the graph corresponding to the equivalent implication. We have a solution if and only if no variable is in the same strongly connected component as it's negation.

#### Extra

Why you shouldn't read to much into t-SNE and how it can read to some fallacies: https://distill.pub/2016/misread-tsne/
