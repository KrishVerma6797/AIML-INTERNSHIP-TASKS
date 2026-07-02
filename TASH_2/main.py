import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv("Titanic-Dataset.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.mean(numeric_only=True))
print(df.median(numeric_only=True))
print(df.mode(numeric_only=True))              
print(df.isnull().sum())

               
plt.figure(figsize=(8,5))
plt.boxplot(df['Fare'])
plt.show()

df.hist(figsize=(8,5))
plt.show()
plt.hist(df["Fare"], bins=20)
plt.show()

plt.boxplt(df['Age'].dropna())

sns.pairplot(['Fair','Age','Pclass'])
plt.show()

corr=df.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.show()

sns.histplot(df['Age'],kde=True)
plt.show()

sns.countplot(x='Survived',data=df)
plt.show()

sns.countplot(x="Sex", hue="Survived", data=df)
plt.show()

sns.countplot(x="Pclass", hue="Survived", data=df)
plt.show()
