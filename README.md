# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Load the dataset.
3.Separate independent variables (X) and dependent variable (y).
4.Split the dataset into training and testing data.
5.Create a Logistic Regression model.
6.Train the model using training data.
7.Predict the placement status using test data.
8.Evaluate the model using accuracy score.
9.Display the result.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DHIVYA DARSHNEE U
RegisterNumber:  21222522027

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'CGPA':[6.5, 7.2, 8.0, 5.9, 7.5, 6.8, 8.5, 5.5, 7.8, 6.0],
    'IQ': [110, 120, 130, 100, 125, 115, 140, 95, 135, 105],
    'Placed': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['CGPA', 'IQ']]
y = df['Placed']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


model = LogisticRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Predicted Placement Status:", y_pred)
print("Accuracy of the model:", accuracy)
print("Confusion Matrix:\n", cm)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model = LogisticRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Predicted Placement Status:", y_pred)
print("Accuracy of the model:", accuracy)
print("Confusion Matrix:\n", cm)

*/
```

## Output:

Predicted Placement Status: [1 1 0]
Accuracy of the model: 1.0
Confusion Matrix:
 [[1 0]
 [0 2]]
Predicted Placement Status: [1 1 0]
Accuracy of the model: 1.0
Confusion Matrix:
 [[1 0]
 [0 2]]


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
