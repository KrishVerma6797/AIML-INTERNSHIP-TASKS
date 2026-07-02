import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("")
print(df.isnull().sum())
#data type..
print(df.d_types())
df[]=df[].fillna().mean()

ohe=OneHotEncoding
cat_col=df.d_type(type='object')
cat_col.ohe
ss=StanderScaler
df.ss()

plt.boxplot()
