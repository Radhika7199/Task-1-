#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[2]:


url= "http://bit.ly/w-data"
data=pd.read_csv(url)
data


# In[3]:


data.plot(x='Hours',y='Scores',style='o',markerfacecolor='blue')
plt.title('Percentage of an student based on the no. of study hours')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.grid(True,color='grey',linestyle='-.')


# In[4]:


sns.regplot(x='Hours',y='Scores',data=data)
plt.grid(True,color='grey',linestyle='-.')


# In[5]:


correlation=data.corr()
sns.heatmap(correlation,annot=True)


# In[22]:


x=data.iloc[:,:1]
y=data.iloc[:,1:]
x


# In[10]:


x_train,x_test,y_train,y_test,=train_test_split(x,y,test_size=0.30,random_state=0)


# In[24]:


print (x_train.shape)
print (y_train.shape)


# In[25]:


regression=LinearRegression()
regression.fit(x_train,y_train)


# In[27]:


regression.coef_


# In[33]:


regression.intercept_


# In[30]:


y_predict=regression.predict(x_test)


# In[39]:


y_predict= pd.DataFrame(y_predict,columns=['Predicted values'])


# In[40]:


y_predict.head()


# In[41]:


print('Test Score')
print(regression.score(x_test,y_test))
print(regression.score(x_test,y_test))


# In[43]:


Hours=9.25
own_predict=regression.predict([[Hours]])
print("no. of hours = {}".format(Hours))
print("no. of hours = {}".format(own_predict[0]))


# In[44]:


print("Mean absolutr error", metrics.mean_absolute_error(y_test,y_predict))


# In[ ]:




