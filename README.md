# Project Portfolio

Hi!

My name is Javid Sultan, I am currently a graduate student at Texas A&M studying Statistics. This GitHub repository contains all of my project, ranging from data visualization, implementing machine learning algorithms, building Neural networks to detect images, etc.

Each project has either a ReadMe which describes the project further or is a Jupyter notebook which will be interactive with commentary and code. A brief summary of each project can be found below. 

If you would like to reach out to me, my email is jsultan100@gmail.com , I hope you enjoy my portfolio!

## **Classification, Regression, and Unsupervised Learning**

### Housing Price Prediction with Penalized Regression and Tree Methods
[Jupyter Notebook](https://github.com/jsultan/Project-Portfolio/blob/master/Regularized%20Linear%20Regression%20and%20Model%20Stacking/Ames_Housing.ipynb)

This project will attempt to predict housing prices in Ames, Iowa by stacking regularized linear methods (LASSO, Ridge, ElasticNet) as well as tree methods (RandomForest and gradient boosting) into one model.

This script has gotten me to the top 20% of the Kaggle competition, and the Jupyter notebook will go more into depth how I achieved this score.

<p align="center">
<img src = "https://github.com/jsultan/Project-Portfolio/blob/master/Regularized%20Linear%20Regression%20and%20Model%20Stacking/saleprice.png" width="300" align = "center" >


### Prediction of Employee Turnover using Ensemble Leaning for Greater Model Performance
[Jupyter Notebook](https://github.com/jsultan/Project-Portfolio/blob/master/Ensemble%20Leaning%20for%20Greater%20Model%20Performance/Ensemble%20Leaning%20for%20Greater%20Model%20Performance.ipynb)

In this notebook we will set out to increase our model performance by stacking multiple classification algorithms to yield a higher accuracy when predicting churn rates of employees. This dataset was generated by IBM in order to predict employee turnover given 40+ features ranging from Education level, salary, commute times, and job satisfaction. The goal of this notebook will not be to create the MOST accurate classifier, but rather a demonstration on how simple stacking and majority voting can improve a model's performance. With this in mind, there will not be as much hyper-parameter tuning or feature engineering as there are in my other notebooks.

<p align="center">
<img src = "https://github.com/jsultan/Project-Portfolio/blob/master/Ensemble%20Leaning%20for%20Greater%20Model%20Performance/heatmap.png" width="500" align = "center" >
</p>

### Visualization of High Dimensional Datasets using t-Distributed Stochastic Neighbor Embedding
[Jupyter Notebook](https://github.com/jsultan/Project-Portfolio/blob/master/t-SNE%20Viz/t-SNE%20Visualization.ipynb)

In this notebook we aim to use the technique known as t-Distributed Stochastic Neighbor Embedding (t-SNE) in order to visualize high dimensional data. This technique created by Geoffrey Hinton and Laurens van der Maaten is a nonlinear dimensionality reduction method which focuses on mapping high dimensional data into two dimensions, which can then be seen through scatterplots. As opposed to PCA, whose objective function determines the linear combination which maximizes overall variance of the dataset, t-SNE focuses to preserve the local distances of the high-dimensional data in some mapping to low-dimensional data.

<p align="center">
<img src = "https://github.com/jsultan/Project-Portfolio/blob/master/t-SNE%20Viz/pic1.png" width="750"/><img src = "https://github.com/jsultan/Project-Portfolio/blob/master/t-SNE%20Viz/pic3.png" width="750"/>
</p>


### Classifier Boundary Visualizations Using Kernel PCA
[Jupyter Notebook](https://github.com/jsultan/Project-Portfolio/blob/master/Visualizing%20Classifier%20Boundaries%20using%20Kernal%20PCA/cancer.ipynb)

This Jupyter notebook will analyze different PCA methods as well as how boundaries are created by different types of classifiers on a dataset describing cancerous tumors. A combination of both parametric and non-parametric methods will be used including: Logistic Regression, Naieve Bayes, K-Nearest Neighbor, Random Forest, and Support Vector Machines (using three different kernels)

<p align="center">
<img src = "https://github.com/jsultan/Project-Portfolio/blob/master/Visualizing%20Classifier%20Boundaries%20using%20Kernal%20PCA/KernelPCA.png" width="500" align = "center" >
</p>

## **Deep Learning**

### Artificial Neural Network Gender Voice Recognition
[Project Overview](https://github.com/jsultan/Project-Portfolio/tree/master/Artificial%20Neural%20Network%20Gender%20Voice%20Recognition)

This voice classification repository will train eight machine learning algorithms, ranging from K-Nearest Neighbors to Feed Forward Neural Networks, in order classify a voice as being male or female. After these algorithms are trained, any .WAV file can be analyzed and tested on to determine if the voice is male or female.

<p align="center">
<img src = "https://github.com/jsultan/Project-Portfolio/blob/master/Artificial%20Neural%20Network%20Gender%20Voice%20Recognition/Figure_5.png" width="500" align = "center" >
</p>

### Recognizing Cats and Dogs using a Convolutional Neural Network
[Jupyter Notebook](https://github.com/jsultan/Project-Portfolio/blob/master/Convolutional%20Neural%20Net%20-%20Dog%20vs%20Cat%20Recognition/CatsAndDogs.ipynb)

The aim of this notebook to is train a CNN to detect if a picture contains a cat or a dog using Keras with a Tensorflow backend. This is one of my first forays into CNN's so this project was intended for me to gain a better understanding of preprocessing, building, and tuning, a convolutional neural network. The data is organized into 8000 pictures evenly split between cats and dogs which is used to train the CNN, and an additional 2000 pictures to use as a validation set. 

<p align="center">
<img src = "https://github.com/jsultan/Project-Portfolio/blob/master/Convolutional%20Neural%20Net%20-%20Dog%20vs%20Cat%20Recognition/catdog.png" width="500" align = "center" >
</p>

### Classifying Handwritten Numbers	using a Convolutional Neural Network
[Project Overview](https://github.com/jsultan/Project-Portfolio/tree/master/Convolutional%20Neural%20Net%20-%20Classifying%20Handwritten%20Numbers)

This script creates a convolution neural network in R for the MNIST dataset (currently a competition in kaggle). This dataset contains 28x28 pixel uploaded greyscale images of handwritten numbers ranging from 0 to 1, with each pixel value ranging from 0 to 255. Each number has minor which the CNN attempts to learn and then recognize on the test dataset.

## **Natural Language Processing**

### Classifying User Consumer Satisfaction Based on Resturant Reviews
[Jupyter Notebook](https://github.com/jsultan/Project-Portfolio/blob/master/NLP%20-%20Classifying%20Restaurant%20Reviews/NLP%20for%20Restaurant%20Reviews.ipynb)

In this Jupyter notebook, we will carry out some natural language processing on a dataset containing reviews of a certain restaurant. The aim of this NLP task is to classify reviews based on if the reviewer liked the restaurant or not. We will create a Bag-Of-Words model which will break down sentences into its constituent terms. It will then analyze the frequency of used terms along with our target variable, whether or not a review liked the restaurant.


### WordCloud Creator and Text Analysis
[Project Overview](https://github.com/jsultan/Project-Portfolio/tree/master/Text%20Analysis%20and%20Auto%20WordCloud%20Generator)

This project was one of the first I undertook to further my understand of R, as well as understanding basic text analysis. 

This script creates some user defined functions which form word clouds and histograms of the most used words in any file. In this case, I used the function on the three main canonical literature pieces of the Abrahamic faiths (Old Testament, New Testament, and Quran)


<p align="center">
<img src = "https://github.com/jsultan/Project-Portfolio/blob/master/Text%20Analysis%20and%20Auto%20WordCloud%20Generator/New%20Testament.png" width="300" align = "center" >
