import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import History 
from keras.optimizers import RMSprop

#Import and split data
data = pd.read_csv('voice.csv')


data.isnull().sum()

print("Total number of labels: {}".format(data.shape[0]))
print("Number of male: {}".format(data[data.label == 'male'].shape[0]))
print("Number of female: {}".format(data[data.label == 'female'].shape[0]))


#Visualize data 10 features at a time
for i in np.arange(1, 11, 1):
    plt.subplot(2,5,i)
    sns.boxplot(x = 'label', y = data.iloc[:,i-1], data = data)
    plt.tight_layout()
    
for i in np.arange(11, 21, 1):
    plt.subplot(2,5,i-10)
    sns.boxplot(x = 'label', y = data.iloc[:,i-1], data = data)
    plt.tight_layout()


#Split features and labels
x = data.iloc[:, :-1].values
y = data.iloc[:,-1].values

#Change labels from categorical to integet
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y) #Male = 1 ----- Female = 0 
                      
#Preprocessess data
scaler = StandardScaler()
x = scaler.fit_transform(x)

#Assess for correlations among features
sns.heatmap(np.corrcoef(x, rowvar = 0))

#Dimensionality reduction
var_explain = []
for i in range(1,16):
    pca = PCA(n_components= i)
    x_pca = pca.fit_transform(x)
    var_explain.append(np.sum(pca.explained_variance_ratio_))

plt.plot(var_explain[:])
plt.xlabel('Number of PCA Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA')
plt.yticks(np.arange(.4,1.05,.05))
plt.xticks(np.arange(1,16,1))
plt.show()

pca = PCA(n_components=10)
x_pca = pca.fit_transform(x)

#Test-train split
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = .1)

#Create Dataframe to hold classifier accuracies
classifier_accuracy = pd.DataFrame(np.zeros((8,2)), columns = ['Classifier', 'Accuracy'])

#Logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

parameters = {'penalty': ['l1','l2'], 'C': np.arange(.1,1.6,.1)}
classifier = GridSearchCV(classifier, parameters, scoring = 'accuracy', verbose = 2, cv = 10)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(classifier.best_params_)
print(metrics.accuracy_score(y_pred,y_test))

classifier_accuracy.iloc[0,0] = 'Logistic Regression'
classifier_accuracy.iloc[0,1] = metrics.accuracy_score(y_pred,y_test)


#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()

parameters = {'n_neighbors': np.arange(1, 11, 1), 'weights': ['distance','uniform']}
classifier = GridSearchCV(classifier, parameters, scoring = 'accuracy', verbose = 2, cv = 10)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(classifier.best_params_)
print(metrics.accuracy_score(y_pred,y_test))

classifier_accuracy.iloc[1,0] = 'KNN'
classifier_accuracy.iloc[1,1] = metrics.accuracy_score(y_pred,y_test)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


print(metrics.accuracy_score(y_pred,y_test))

classifier_accuracy.iloc[2,0] = 'Naive Bayes'
classifier_accuracy.iloc[2,1] = metrics.accuracy_score(y_pred,y_test)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

parameters = {'n_estimators': np.arange(100, 1100, 100), 'criterion': ['gini','entropy']}
classifier = GridSearchCV(classifier, parameters, scoring = 'accuracy', verbose = 2, cv = 10)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(classifier.best_params_)
print(metrics.accuracy_score(y_pred,y_test))

classifier_accuracy.iloc[3,0] = 'Random Forest'
classifier_accuracy.iloc[3,1] = metrics.accuracy_score(y_pred,y_test)

#XGBoost
from xgboost import XGBClassifier
from xgboost import plot_tree
classifier = XGBClassifier()

parameters = {'n_estimators': np.arange(100, 1100, 100)}
classifier = GridSearchCV(classifier, parameters, scoring = 'accuracy', verbose = 2, cv = 10)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(classifier.best_params_)
print(metrics.accuracy_score(y_pred,y_test))

classifier_accuracy.iloc[4,0] = 'XGBoost'
classifier_accuracy.iloc[4,1] = metrics.accuracy_score(y_pred,y_test)

#Support Vector - Linear
classifier = SVC(kernel = 'linear')

parameters = {'C':np.arange(.1,1.5,.05)}

classifier = GridSearchCV(classifier, parameters, scoring = 'accuracy', verbose = 3, cv = 10)
classifier.fit(x_train, y_train)

print(classifier.best_params_)
print(metrics.accuracy_score(y_pred,y_test))

y_pred = classifier.predict(x_test)

classifier_accuracy.iloc[5,0] = 'SVM - Linear'
classifier_accuracy.iloc[5,1] = metrics.accuracy_score(y_pred,y_test)

#Support Vector - GuassianRBF
classifier = SVC(kernel = 'rbf')

parameters = {'C':np.arange(.1,1.5,.05), 'gamma': np.arange(.1,.5, .01) }

classifier = GridSearchCV(classifier, parameters, scoring = 'accuracy', verbose = 3, cv = 10)
classifier.fit(x_train, y_train)

print(classifier.best_params_)

y_pred = classifier.predict(x_test)

classifier_accuracy.iloc[6,0] = 'SVM - rbf'
classifier_accuracy.iloc[6,1] = metrics.accuracy_score(y_pred,y_test)


#Build Neural Network 
n_cols = x_train.shape[1]
y_train_nn = to_categorical(y_train, 2)
y_train_nn = y_train_nn[:,1]

hist = History()

model = Sequential()

model.add(Dense(1500, kernel_initializer = 'uniform', activation='relu', input_dim = n_cols))
model.add(Dropout(.1))

model.add(Dense(1500, kernel_initializer = 'uniform', activation='relu'))
model.add(Dropout(.1))

model.add(Dense(1500, kernel_initializer = 'uniform', activation='relu'))
model.add(Dropout(.1))


model.add(Dense(1, activation='sigmoid'))

rms = RMSprop(lr = .0005)

model.compile(optimizer=rms, 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.fit(x_train, y_train_nn, batch_size = 20, epochs = 40, validation_split = .1, callbacks = [hist])

y_pred_nn =  model.predict(x_test)
y_pred_nn = np.round(y_pred_nn)

print(metrics.accuracy_score(y_pred_nn,y_test))
print(metrics.classification_report(y_test, y_pred_nn))


plt.plot(hist.history['acc'], color = 'red')
plt.plot(hist.history['val_acc'], color = 'blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

classifier_accuracy.iloc[7,0] = 'Neural Network'
classifier_accuracy.iloc[7,1] = metrics.accuracy_score(y_pred_nn,y_test)

classifier_accuracy_sorted = classifier_accuracy.sort_values('Accuracy', ascending = False)

plt.barh(np.arange(7, -1, -1), width = 'Accuracy',  data = classifier_accuracy_sorted, align = 'center', alpha = .5)
plt.yticks(np.arange(7, -1, -1), classifier_accuracy_sorted.Classifier, rotation = 30)
plt.xlim([.95,1])
plt.ylabel('Classification Algorithim')
plt.xlabel('Test Set Accuracy')
plt.title('Comparison of Classification Algorithims')
plt.show()

