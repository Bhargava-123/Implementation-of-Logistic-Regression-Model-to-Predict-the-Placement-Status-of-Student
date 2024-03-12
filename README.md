# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the dataset and necessary libraries
2. Identify the target variable and convert the independent categorical variables to numerical data using LabelEncoding
3. Split train, test dataset and train the Logistic Regression model to find the optimum threshold of the target variable
4. Display the confusion matrix to assess the accuracy of the model.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Bhargava S
RegisterNumber: 212221040029
import pandas as pd

data   = pd.read_csv("./datasets/Placement_Data.csv")
data.head()

data1 = data.copy()
data1.head()

data1 =  data1.drop(['sl_no','salary'],axis=1)

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le  = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

x = data1.iloc[:,:-1]
x

y = data1["status"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  =train_test_split(x,y,test_size = 0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy = accuracy_score(y_test,y_pred)
confusion = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
print("Accuracy Score:",accuracy)
print("Confusion Matrix: ",confusion)
print("Classification Report:",cr)

from sklearn import metrics
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
*/
```

## Output:
![image](https://github.com/Bhargava-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/85554376/93124bc0-cf1e-4ed0-9680-1bf45eb7bcf2)

![image](https://github.com/Bhargava-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/85554376/aa0f0f5b-1fd3-46ca-b26f-cd2915b29a8c)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
