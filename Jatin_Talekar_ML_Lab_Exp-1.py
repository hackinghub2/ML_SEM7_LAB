#!/usr/bin/env python
# coding: utf-8

# Name : Jatin Prashant Talekar
# Roll No : 21102B0055
# BE CMPN B
# B3

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('housing.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df=df.dropna()


# In[7]:


df.isnull().sum()


# In[8]:


df.isna().sum()


# In[9]:


df.dtypes


# In[10]:


plot=sns.pairplot(df)


# In[11]:


import math


# In[12]:


x_plot=df['housing_median_age']
y_plot=df['median_house_value']
x = pow(x_plot,1)
y = pow(y_plot,-1/3)
plot=plt.scatter((x),(y),color="magenta")


# In[13]:


df1=df.drop(columns=['ocean_proximity','latitude','longitude'])


# In[14]:


correlation_matrix = df1.corr()
sns.heatmap(correlation_matrix, cmap='inferno', annot=True)


# In[16]:


x=df['median_house_value']
y=df['ocean_proximity']
fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(y, x, color ='yellow',
 width = 0.4)
plt.xlabel("ocean_proximity")
plt.ylabel("median_house_value")
plt.title("Bar chart")
plt.show()


# In[19]:


df2=pd.DataFrame(df, columns=['ocean_proximity','median_house_value'])
df2["rank"] = df2.groupby(['ocean_proximity'])["median_house_value"].rank("dense", ascending=False)

df2[df2["rank"]==1.0][['ocean_proximity','median_house_value']]


# In[20]:


df2.groupby(['ocean_proximity'])['median_house_value'].max()


# In[21]:


X = df1.drop('median_house_value',axis= 1)
y = df1['median_house_value']
print(X)
print(y)


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# In[23]:


# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[24]:


# creating a regression model
model = LinearRegression()


# In[25]:


model.fit(X_train,y_train)


# In[26]:


predictions = model.predict(X_test)
predictions


# In[27]:


print(
  'mean_squared_error : ', mean_squared_error(y_test, predictions))
print(
  'mean_absolute_error : ', mean_absolute_error(y_test, predictions))


# In[28]:


plt.scatter(range(len(y_test)), y_test, color='orange')
plt.scatter(range(len(predictions)), predictions, color='green')

plt.show()


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns

X1 = X.sample(500, random_state=1)
y1 = y.sample(500, random_state=1)

plt.figure(figsize=(18,12))
sns.regplot(x=X1['median_income'], y=y1, data=df1, color='red')  # Change 'blue' to the desired color
plt.show()


# In[30]:


train_score =model.score(X_train,y_train)
test_score = model.score(X_test,y_test)

print('Linear regression score: \n')
print('Train score : ',round(train_score*100),'%')
print('Test score : ',round(test_score*100),'%')


# In[31]:


max_pred=predictions.max()
max_pred


# In[32]:


r2=r2_score(y_test,predictions)


# In[33]:


r2


# In[ ]:




