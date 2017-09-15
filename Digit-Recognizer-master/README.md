# Digit-Recognizer

This script creates a convolution neural network in R for the MNIST dataset (currently a competition in kaggle). This dataset contains 28x28 pixel uploaded greyscale images of handwritten numbers ranging from 0 to 1, with each pixel value ranging from 0 to 255. Each number has minor which the CNN attempts to learn and then recognize on the test dataset.

This script borrows heavily from the MXNet tutorial page and the following kaggle kernel:
http://mxnet.io/tutorials/python/mnist.html

https://www.kaggle.com/srlmayor/easy-neural-network-in-r-for-0-994

With enough number of rounds for this CNN you will get an accuracy of ~.993 which will get you to the top 15% in the Kaggle competition
