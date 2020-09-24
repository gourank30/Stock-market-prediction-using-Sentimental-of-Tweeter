import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
from datetime import datetime

company_list = ['aapl','ibm', 'goog']
c_name=["APPLE", "IBM", "GOOGLE"]
company=pd.read_csv('data.csv')
#Adj Close
plt.figure(figsize=(12, 8))
plt.subplots_adjust(hspace=0.77)



for i in range(3):
    plt.subplot(2, 2, i+1)
    data=company[company.company_name==company_list[i]]
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index(data['Date'], inplace=True)
    data['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"{c_name}")
plt.show()



#Volume
plt.figure(figsize=(12, 8))
plt.subplots_adjust(hspace=0.77)
for i in range(3):
    plt.subplot(2, 2, i+1)
    data=company[company.company_name==company_list[i]]
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index(data['Date'], inplace=True)
    data['Volume'].plot()
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"{c_name[i]}")
plt.show()



#High
plt.figure(figsize=(12, 8))
plt.subplots_adjust(hspace=0.77)
for i in range(3):
    plt.subplot(2, 2, i+1)
    data=company[company.company_name==company_list[i]]
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index(data['Date'], inplace=True)
    data['High'].plot()
    plt.ylabel('High')
    plt.xlabel(None)
    plt.title(f"{c_name[i]}")
plt.show()


#Low
plt.figure(figsize=(12, 8))
plt.subplots_adjust(hspace=0.77)
for i in range(3):
    plt.subplot(2, 2, i+1)
    data=company[company.company_name==company_list[i]]
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index(data['Date'], inplace=True)
    data['Low'].plot()
    plt.ylabel('Low')
    plt.xlabel(None)
    plt.title(f"{c_name[i]}")
plt.show()





