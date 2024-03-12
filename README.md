# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M.Harini
RegisterNumber: 212222240035
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('/content/student_scores.csv')
df.head()

#segregating data to variables
x = df.iloc[:, :-1].values
x

#splitting train and test data
y = df.iloc[:, -1].values
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## Output:
df.head()

![image](https://github.com/Harinimuthu17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/130278614/9cc39e77-6102-4a63-8516-7569d490d41e)


df.tail()

![image](https://github.com/Harinimuthu17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/130278614/dd618901-f9ac-4b44-a30a-6561fcc71c00)


Array value of X

![image](https://github.com/Harinimuthu17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/130278614/7f84be88-6c9a-42e9-aee8-09037bd4f34c)


Array value of Y

![image](https://github.com/Harinimuthu17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/130278614/80f403ee-3e0e-4e02-a891-46998d2426f9)

Values of y preidction

![image](https://github.com/Harinimuthu17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/130278614/bc229b6d-ad0b-4f1f-bf35-7b58a900e4a4)


Array values of Y test


![image](https://github.com/Harinimuthu17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/130278614/b24be7cc-93be-4d4b-bfec-66d22c47bae4)

Training and Testing set


![image](https://github.com/Harinimuthu17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/130278614/41ce1586-2c7d-4735-beac-dff5de095b71)

![image](https://github.com/Harinimuthu17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/130278614/ff11332e-154c-40a5-99c9-45c52f6a5aa2)


Values of MSE,MAE and RMSE

![image](https://github.com/Harinimuthu17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/130278614/c3d47156-bc56-4979-86d0-0e58ccac0974)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

 



