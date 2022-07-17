#!/usr/bin/env python
# coding: utf-8

# # Background
# - Searching for raw datasets
# - Cleaning data
# - Proposed questions 

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[57]:


a = np.array([7,8,9], ndmin = 1)


# In[28]:


import pandas as pd
df= pd.read_csv("C:/Users/admin/Documents/Sales_data.csv")
df


# In[29]:


df.isnull().sum()


# In[30]:


df.loc[df.isnull().any(axis=1)]


# In[31]:


df = df.dropna().reset_index(drop = True)
df


# In[32]:


df.isnull().sum()


# In[33]:


duplicateddf = df[df.duplicated()]
print(duplicateddf)


# In[34]:


df.drop_duplicates()


# In[35]:


df = df.drop_duplicates().reset_index(drop = True)
df


# In[36]:


df.info()


# In[37]:


from datetime import datetime

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df.info()


# In[38]:


df


# In[39]:


df["Total Profit"] = df["Total Revenue"] - df["Total Cost"]
df


# In[40]:


df['Year'] = pd.DatetimeIndex(df['Order Date']).year
df


# In[41]:


df.round()


# In[42]:


df.describe()


# # Data Visualisation

# In[43]:


#Barchart analysis of Sales Channel
Saleschannel_profit=df.groupby('Sales Channel')['Total Profit'].sum().to_frame().reset_index()
plt.bar(Saleschannel_profit['Sales Channel'], Saleschannel_profit['Total Profit'], color = ["blue","Red"])
plt.title('Total Profit per Sales Channel')
plt.xlabel('Sales Channel')
plt.ylabel('Total Profit in millions') 
plt.show()


# In[50]:


logical_condition <- c(1:10)
logical_condition[(logical_condition>5)]


# In[55]:


import datetime from datetime
datetime.today()+10


# In[44]:


#Sale volume per Year
df_pivot = pd.pivot_table(df, values="Units Sold",index="Year",columns="Sales Channel")
ax = df_pivot.plot(kind="bar",alpha=1.0,color= ['blue','red'] )
plt.title('Sales volume per Item Type')
plt.ylabel('Sales volume') 
plt.xlabel(None)
plt.show()


# In[ ]:





# In[45]:


#Profit per Item type
df_pivot = pd.pivot_table(df, values="Total Profit",index="Item Type",columns="Sales Channel")
ax = df_pivot.plot(kind="bar",alpha=1.0,color= ['blue','red'] )
plt.title('Total Profit per Item Type')
plt.ylabel('Total Profit in millions') 
plt.xlabel(None)
plt.show()


# In[46]:


#Item type analysis
value = df['Item Type'].value_counts()
my_colors = ['orange', 'royalblue', 'yellow', 'green', 'red','purple','darkblue','grey','maroon','pink','lightgreen', 'turquoise']
my_explode = (1, 0.1, 1)
fig1, ax1 = plt.subplots(figsize=(14, 8))
ax1.pie(value.values,labels=value.index,autopct='%1.1f%%', colors=my_colors)
plt.title('Item Type')
plt.show()


# In[47]:


#Regional analysis
Revenue_region=df.groupby('Region')['Total Revenue'].sum().to_frame().reset_index()
plt.barh(Revenue_region['Region'], Revenue_region['Total Revenue'],color = ['green'])
plt.title('Total Revenue per Region')
plt.xlabel('Total Revenue')
plt.ylabel('Region') 
plt.show()


# In[48]:


df.groupby('Year')['Total Profit'].sum()


# In[49]:


#Total profit per year analysis
import pandas as pd
import matplotlib.pyplot as plt

Data = {'Year': [2010,2011,2012,2013,2014,2015,2016],
        'Total Profit': [2.412051e+08,2.577715e+08,2.756043e+08,2.687148e+08,2.544421e+08,2.725062e+08,2.538318e+08,]
       }
  
df = pd.DataFrame(Data,columns=['Year','Total Profit'])
plt.plot(df['Year'], df['Total Profit'], color='darkblue', marker='o')
plt.title('Total Profit per Year', fontsize=14)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Total Profit in hundred millions USD', fontsize=10)
plt.grid(color = 'darkblue', linestyle = '--', linewidth = 0.5)
plt.grid(True)
plt.show()

