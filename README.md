# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


 1. Import the required libraries and read the dataframe.
 2. Assign hours to X and scores to Y.
 3. Implement training set and test set of the dataframe
 4.  Plot the required graph both for test data and training data.
 5. Find the values of MSE , MAE and RMSE.
 


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M.Harini
RegisterNumber: 212222240035
*/
import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset)

# assigning hours to X & Scores to Y
X=dataset.iloc[:,:1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![image](https://user-images.githubusercontent.com/119389139/230030388-5cc59fa1-179f-40a1-b039-bd91ee6b04f1.png)

![image](https://user-images.githubusercontent.com/119389139/230030496-11aee79a-91f1-4641-88a3-9a63ef1742e3.png)

![image](https://user-images.githubusercontent.com/119389139/230030599-fdde8425-5b46-425e-a4b9-45cd5d363436.png)

![image](https://user-images.githubusercontent.com/119389139/230030817-fcb96383-b463-4ac3-a5fc-ac5dfcbf5721.png)

![image](https://user-images.githubusercontent.com/119389139/230031140-bf4c46e9-dc2f-4cc4-b23c-37d6d73fcc62.png)

![image](https://user-images.githubusercontent.com/119389139/230033037-d05b409c-8022-48b9-b423-a116b81090b5.png)

![image](https://user-images.githubusercontent.com/119389139/230034346-339ed1aa-9572-4a0b-aca6-5fbf458cded9.png)

![image](https://user-images.githubusercontent.com/119389139/230033991-1ca0ee04-3ca5-4dde-a0bd-db2935d38122.png)

![image](https://user-images.githubusercontent.com/119389139/230033345-669e2734-ae50-4242-92f4-fa526a6fd3df.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
