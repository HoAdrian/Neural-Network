# Author 
Name: Shing Hei Ho

Github ID: HoAdrian

Date: 15 March 2022

# Purpose
This is my first attempt to understand and implement a neural network rigorously. 
To prepare for this, I have learned multivariable calculus and linear algebra. This
project took me two days. 

# Code contents
This is a neural network library modified from the basic code provided by
Michael Nielsen. I modified this code to employ object-oriented approach, which makes
the code more organized, understandable and cleaner.

# learning outcomes
First, I worked out the maths of representing neurons, feeding forward, stochastic gradient descent and back
propagation. Then, I implemented it in Python with the aid of the example code provided by Michael Nielsen, 
which was a new language for me. 

I have not figured how to download and process a dataset, like a set of hand-written digit images with labels.
However, I have learned to convert the mathematics of neural network into actual code (the most challenging)
and use matrices to keep track of tons of neurons. 

I found a layer of activation can be represented as a matrix in which each column is a 
sample of the mini batch with slight modification on the code, which will make the code 
cleaner and probably more efficient; however, I haven't implemented this feature since
I need to know how to process a dataset before implementing this.

# References
1. 2012-2018 Michael Nielsen - http://neuralnetworksanddeeplearning.com/index.html
