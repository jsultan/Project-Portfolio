# Voice-Classifier

## Goal
This voice classification repository will train eight machine learning algorithms, ranging from K-Nearest Neighbors to Feed Forward Neural Networks, in order classify a voice as being male or female. After these algorithms are trained, any .WAV file can be analyzed and tested on to determine if the voice is male or female.

## The Data
The dataset was found through Kaggle's open data repository, which is linked at the bottom. The R code (sourced from another Github portfolio) analyzes a .WAV file to extract the following features:

- **meanfreq**: mean frequency (in kHz)
- **sd**: standard deviation of frequency
- **median**: median frequency (in kHz)
- **Q25**: first quantile (in kHz)
- **Q75**: third quantile (in kHz)
- **IQR**: interquantile range (in kHz)
- **skew**: skewness (see note in specprop description)
- **kurt**: kurtosis (see note in specprop description)
- **sp.ent**: spectral entropy
- **sfm**: spectral flatness
- **mode**: mode frequency
- **centroid**: frequency centroid (see specprop)
- **peakf**: peak frequency (frequency with highest energy)
- **meanfun**: average of fundamental frequency measured across acoustic signal
- **minfun**: minimum fundamental frequency measured across acoustic signal
- **maxfun**: maximum fundamental frequency measured across acoustic signal
- **meandom**: average of dominant frequency measured across acoustic signal
- **mindom**: minimum of dominant frequency measured across acoustic signal
- **maxdom**: maximum of dominant frequency measured across acoustic signal
- **dfrange**: range of dominant frequency measured across acoustic signal
- **modindx**: modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range

This will create a csv file which we will use in Python to train the SVM and NN. A quick EDA of these features with respect to gender can be seen here:

![](https://github.com/jsultan/Project-Portfolio/blob/master/Artificial%20Neural%20Network%20Gender%20Voice%20Recognition/Figure_1.png)

![](https://github.com/jsultan/Project-Portfolio/blob/master/Artificial%20Neural%20Network%20Gender%20Voice%20Recognition/Figure_2.png)

There seems to be a large difference in averages and IQR's for almost all the features. Also, the dataset is balanced, with half the training sample voices male and the other female, so we will be using model accuracy as the metric to determine the best algorithm.

Despite there only being 20 features , we can see that some features are highly correlated. We will do PCA to reduce the dimensions of the dataset in order to train the algorithms faster. We will see that halving the amount dimensions will still retain ~97% of the explained variance.

![](https://github.com/jsultan/Project-Portfolio/blob/master/Artificial%20Neural%20Network%20Gender%20Voice%20Recognition/Figure_3.png)

![](https://github.com/jsultan/Project-Portfolio/blob/master/Artificial%20Neural%20Network%20Gender%20Voice%20Recognition/PCA.png)

All the algorithm’s hyperparameters were tuned using a ten-fold cross validation grid search. Scikit-Learn's GridSearchCV came in handy for this as we were able to test a wide variety of values for each algorithim. The final accuracies shown below were calculated using a holdoff test set that was split from the training set in the beginning. It was important to do so in order to ensure that the algorithms were not trained using any of the test data.  

![](https://github.com/jsultan/Project-Portfolio/blob/master/Artificial%20Neural%20Network%20Gender%20Voice%20Recognition/Figure_5.png)

The top results were in line with what I expected, with the radial kernel SVM and Neural network performing similarly. I was a bit surprised that the KNN algorithm out performed both the random forest and gradient boosted tree, however there is a wider variety of hyperparamters that can be tuned going forward. Overall I was very happy with >99% accuracy of the SVM and NN, so I went forward with testing new .WAV files against these classifiers.





# Going Forward
There is a second python script called voice_test which will allow you to classify newly created test data after the neural network/SVM has been trained. I've used it on myself, my wife, other family members, as well as musicians whose voices are androgynous. So far the neural network has classified all these correctly.

## Links to dataset and R Code
https://www.kaggle.com/primaryobjects/voicegender
https://github.com/RufoJ/Specan
