
import pandas as pd
import numpy as np
import re





ratings = pd.read_csv(r"C:\Users\user\Desktop\AMZN2.csv")
df=ratings.loc[:,['Open','Close']]
print(df.head())
ratings['Date'] = pd.to_datetime(ratings['Date'])
ratings['date'] = ratings['Date'] - pd.to_timedelta(ratings['Day'], unit='d')
df1=ratings.loc[:,['date']]
print(df1.head)
df1.to_csv(r'C:\Users\user\Desktop\NewDate.csv',index=False)


