#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import pandas
import pandas as pd

#  Load the dataset
file_path = r"C:\Users\mwaki\Downloads\1569583467_hour\hour.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())


# In[4]:


# check for null values and drop records with NAs
df.isnull().sum()

# Drop records wtih NAs
df.dropna(inplace=True)


# In[6]:


# Sanity Checks
# Check if 'registered + casual = cnt' for all records
df = df[df['registered'] + df['casual'] == df['cnt']]

# Ensure month values =are 1-12
df = df[df['mnth'].between(1, 12)]

# Ensure hour values are 0-23
df = df[df['hr'].between(0,23)]


# In[7]:


# Drop 'casual', 'registered', and 'dteday'
inp1 = df.drop(['casual', 'registered', 'dteday'], axis=1)


# In[11]:


# Univariate Analysis

# Import matplotlib.plypl;ot
import matplotlib.pyplot as plt
# Import seaborn
import seaborn as sns

# Describe numberical fields
inp1.describe()

# Density plot for temp
sns.kdeplot(inp1['temp'])
plt.show()

# Boxplot for atemp
sns.boxplot(x=inp1['atemp'])
plt.show()

# Histogram for hum
inp1['hum'].hist()
plt.show()

# Density plot for windspeed
sns.kdeplot(inp1['windspeed'])
plt.show()

# Box and density plot for cnt
sns.boxplot(x=inp1['cnt'])
plt.show()
sns.kdeplot(inp1['cnt'])
plt.show()


# In[13]:


# Outlier Treatment for 'cnt'
# Find percentiles
percentiles = inp1['cnt'].quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95,0.99])

# Decide on a cutoff percentile and drop records above it
cutoff = percentiles[0.99]
inp2 = inp1[inp1['cnt'] <= cutoff]


# In[14]:


# Biviate analysis

# Boxplot for 'cnt' vs. 'hour'
sns.boxplot(x='hr', y='cnt', data=inp2)
plt.show()
# Boxplot for 'cnt' vs. 'weekday'
sns.boxplot(x='weekday', y='cnt', data=inp2)
plt.show()
# Boxplot for 'cnt' vs. 'month'
sns.boxplot(x='mnth', y='cnt', data=inp2)
plt.show()
# Boxplot for 'cnt' vs. 'season'
sns.boxplot(x='season', y='cnt', data=inp2)
plt.show()
# Bar plot with median value of 'cnt' for each 'hr
inp2.groupby('hr')['cnt'].median().plot(kind='bar')
plt.show()
# Correlation matrix for 'atemp', 'temp', 'hum'. and 'windspeed'
inp2[['atemp', 'temp', 'hum', 'windspeed']].corr()


# In[16]:


# Data Preprocessing

# Create copy before making modifications
inp2 = inp2.copy()
# Treat 'mnth' column
inp2.loc[inp2['mnth'].isin([5, 6, 7, 8, 9, 10]), 'mnth'] = 5
# Treat ;hr column
inp2.loc[:, 'hr'] = inp2['hr'].apply(lambda x: 0 if 0 <= x <= 5 else (11 if 11 <= x <= 15 else x))
# Get dummies for categorical variables
inp3 = pd.get_dummies(inp2, columns=['season', 'weathersit', 'weekday', 'mnth', 'hr'], drop_first=True)


# In[17]:


# Train Test Split
from sklearn.model_selection import train_test_split

X = inp3.drop('cnt', axis=1)
y = inp3['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[19]:


# Model Building

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Model building
lr = LinearRegression()
lr.fit(X_train, y_train)

# R2 on the train set
y_train_pred = lr.predict(X_train)
print(f"R2 on train set: {r2_score(y_train, y_train_pred)}")

# Predictions on the test set
y_test_pred =lr.predict(X_test)
print(f"R2 on test: {r2_score(y_test, y_test_pred)}")


# In[ ]:


# Make predictions on the test set
# Both the training model and test model have an R2 value is mid-high,  R2 = 0.67 for the 
# training model and R2 = 0.66 for the test model.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




