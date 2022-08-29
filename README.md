# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction. 

## Neural Network Model



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
``` python3
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("data1.csv")

df.head()

x=df[["Input"]].values

y=df[["Output"]].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

import tensorflow as tf

model=tf.keras.Sequential([tf.keras.layers.Dense(8,activation='relu'),
                           tf.keras.layers.Dense(16,activation='relu'),
                           tf.keras.layers.Dense(1)])
model.compile(loss="mae",optimizer="adam",metrics=["mse"])

history=model.fit(x_train,y_train,epochs=1000)

import numpy as np

x_test

preds=model.predict(x_test)
np.round(preds)

tf.round(model.predict([[20]]))

pd.DataFrame(history.history).plot()

r=tf.keras.metrics.RootMeanSquaredError()
r(y_test,preds)

```
## Dataset Information
![data](https://user-images.githubusercontent.com/75234983/187186035-676b656b-16a3-493d-a6d3-1c0d5f87c102.jpg)


## OUTPUT

### Training Loss Vs Iteration Plot
![graph](https://user-images.githubusercontent.com/75234983/187186133-b114f5ac-3031-4a97-98cf-aaa7034f0781.jpg)


### Test Data Root Mean Squared Error
![mean](https://user-images.githubusercontent.com/75234983/187186160-9b118651-7062-4a2b-bdbf-567076343895.jpg)



### New Sample Data Prediction
![sample](https://user-images.githubusercontent.com/75234983/187186212-2a3550b3-61fe-4309-ad08-50fa1857c187.jpg)


## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully
