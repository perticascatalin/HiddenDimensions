# Dragos-Alexandru Rebegea - Summer internship at RIST

I`m a 3rd year student in the Faculty of Mathematics and Computer Science at Babeş-Bolyai University and last summer I had an internship at Romanian Institute of Science and Technology. Here I will document what I learned during this internship.

## Before internship

Because this was going to be my first touch with the machine learning field I started by researching some information about the structure and the syntax of Tensorflow. I found most of what I have learned on Aurélien Geron`s repository that walks through the fundamentals of Machine Learning and Deep Learning using Scikit-Learn and TensorFlow.
https://github.com/ageron/handson-ml

### Introduction to Recurrent Neural Networks
First of all, at the Institut, the purpose was to create a RNN network for text generation, in order to later use it to generate HtML code.
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
I started by reading one of Andrej Karpathy's blog posts where he is talking about the effectiveness and versatility of RNNs. He showed that a simple network of standard RNN cells can generate any type of text: Shakespeare, Latex code, C++ code and basically almost any programming language.

![](http://karpathy.github.io/assets/rnn/latex3.jpeg)

On the same site I found a minimal character-level language model with a Vanilla Recurrent Neural Network, which helped me understand better how a RNN work by replicating this on my own.

For acquiring a deep understanding I looked through Aurélien Geron`s book called "Hands on Machine Learning with Scikit Learn and Tensorflow" where my main goal was not only to learn more about RNN, but to also gain a perspective of another things that may help me in the future. So I learned more about classification,training models and the most , LSTM Cell. Furthermore, reading this book I discovered the main disadvantage of the RNN concept: the vanishing gradient.
### Vanishing gradient

The vanishing gradient affects the RNNs because these networks are used to model time-dependent data, like words in a sentence or in our case it can be tags,special characters like "<",">" or any other words. 

If we think about each timestep as a layer, with weights going from one timestep to the next, we can see that our network will be at least as deep as the number of timesteps. When it comes to sentences, paragraphs, these sequences we’re feeding in can be very long, so the weights at the beginning of the network change less and less, and the RNN becomes worse at modeling **long-term dependencies**. If we’re predicting words of a sentence, the first word in the sentence might actually be really important context for predicting a word at the end, so we don’t want to lose that information.
 That`s why I decided that it would be a better solution to focus on LSTM rather than RNN.
### Third week
http://harinisuresh.com/2016/10/09/lstms/

Knowing that I would use LSTM cells, I read more about why it is more appropriate for sequence working and generating text. 
The main difference between RNN and LSTM is the structure itself. LSTM has some 'gates' that every imput has to get through.
![](http://harinisuresh.com/img/lstm-images/details.png)
|The forget gate |   The input gate  |   The output gate |
| :---: | :---: | :---: |
|controls which parts of the long-term state should be erased    |  controls which parts should be added to the long-term state. |controls which parts of the long-term state should be read and output at this time step  |
Shortly, an LSTM cell can learn to **recognize an important input** (that’s the role of the input gate), **store it** in the long-term state, learn to **preserve** it for as long as it is needed (that’s the role of the forget gate), and learn to **extract** it whenever it is needed.

![](http://harinisuresh.com/img/lstm-images/output.gif)


### Implementing a network with LSTM
Using a toy dataset from the Institute,i implemented an **LSTM** network on word-level using Tensorflow.

incheiere + rezultate ( cause toy dataset)

legatura

despre auto-encoder

string sum

explicatie auto-encoder



