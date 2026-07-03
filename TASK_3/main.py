import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df=pd.read_csv("Titanic-Dataset.csv")
print(df.head())
print(df.isnull().sum())
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop("Cabin", axis=1)


cat_col=df.select_dtypes(include='object').columns
df=pd.get_dummies(df,columns=cat_col,drop_first=True)


scaler=StandardScaler()
nums=['Fare','Age']
df[nums]=scaler.fit_transform(df[nums])
x = df[["Fare"]]
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)
print(mae)
print(mse)

plt.scatter(x_test["Fare"], y_test)
plt.plot(x_test["Fare"], y_pred, color="red")
plt.xlabel("Fare")
plt.ylabel("Survived")
plt.show()

print("Intercept:", lr.intercept_)
print("Coefficients:")
for feature, coef in zip(x.columns, lr.coef_):
    print(feature, ":", coef)
