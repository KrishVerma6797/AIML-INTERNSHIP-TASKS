import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)



df=pd.read_csv("Titanic-Dataset.csv")
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop("Cabin", axis=1)

cat_cols=pd.select_dtypes(include='object').columns
df=pd.get_dummies(df,columns=cat_cols,drop_first=True)

nums=['Fare','Age']
scaler=StandardScaler()
df[nums]=scaler.fit_transform(df[nums])


x=df.drop('Survived',axis=1)
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


cm = confusion_matrix(y_test, y_pred)
print(cm)

prec = precision_score(y_test, y_pred)
print("Precision:", prec)

rec = recall_score(y_test, y_pred)
print("Recall:", rec)

prob = lr.predict_proba(x_test)[:, 1]
roc = roc_auc_score(y_test, prob)
print("ROC-AUC:", roc)

fpr, tpr, threshold = roc_curve(y_test, prob)

plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], "--")
plt.show()

prob = lr.predict_proba(x_test)[:,1]
y_pred = (prob >= 0.5).astype(int)
