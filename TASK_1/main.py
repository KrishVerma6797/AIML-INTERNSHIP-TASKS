import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())
print(df.describe())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop("Cabin", axis=1)
print(df.isnull().sum())



cat_col=df.select_dtypes(include='object').columns
print(cat_col)
df = pd.get_dummies(df, columns=cat_col, drop_first=True)

ss=StanderScaler
nums=df['Age','Fare']
df[nums]=ss.fit_transform(df[nums])

plt.figure(figsize=(12,12))
plt.boxplot(df['Fare'])
plt.show()

plt.figure(figsize=(8,5))
plt.boxplot(df["Age"])
plt.title("Age Boxplot")
plt.show()


q1=df['Fare'].quantile(0.25)
q2=df['Fare'].quantile(0.75)
iqr=q2-q1
lower=q1-iqr*1.5
upper=q2+iqr*1.5
df=df[(df['Fare']>=lower) & (df['Fare']<upper)]
print(df.shape)

df.to_csv("cleaned_titanic.csv", index=False)
