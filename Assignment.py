#!/usr/bin/env python
# coding: utf-8

# Step 1 - Import the library

# In[34]:


import numpy as np
import pandas as pd


# In[36]:


detail ='C:\\Users\\Nivas\\Downloads\\detail.csv'
df=pd.read_csv(detail)
df.head(10)


# Step 2 - Setting up the Data
# This dataset is not bias so we are making it bias for better understanding of the functions

# In[37]:


df.Status = pd.factorize(df.Status)[0]


# In[38]:


X=df.iloc[:,0:12]
y = df['Status']

y = np.where((y == 0), 0, 1)
print("Viewing the imbalanced target vector:\n", y)


# Step 3 - Downsampling the dataset
# First we are selecting the rows where target values are 0 and 1 in two different objects and then printing the number of observations in the two objects.

# In[39]:


w_class0 = np.where(y == 0)[0]
w_class1 = np.where(y == 1)[0]

n_class0 = len(w_class0) 
n_class1 = len(w_class1)
 

print("n_class0: ", n_class0)
print("n_class1: ", n_class1)


# In the output we will see the number of samples having target values as 1 are much more greater than 0. So in downsampling we will randomly select the number of rows having target as 1 and make it equal to the number of rows having taregt values 0.
# Then we have printed the joint dataset having target class as 0 and 1.
# 

# In[40]:


w_class1_downsampled = np.random.choice(w_class1, size=n_class0, replace=False)

print(); print(np.hstack((y[w_class0], y[w_class1_downsampled])))


# In[49]:


detailTemp ='C:\\Users\\Nivas\\Downloads\\detailTemp.csv'
df1=pd.read_csv(detailTemp)
df1.head(10)


# In[50]:


df1.Step_Name = pd.factorize(df1.Step_Name)[0]


# In[52]:


X = df1.iloc[:,0:6]
y = df1['Step_Name']

y = np.where((y == 0), 0, 1)
print("Viewing the imbalanced target vector:\n", y)


# In[53]:


w_class0 = np.where(y == 0)[0]
w_class1 = np.where(y == 1)[0]

n_class0 = len(w_class0) 
n_class1 = len(w_class1)
 

print("n_class0: ", n_class0)
print("n_class1: ", n_class1)


# In[54]:


w_class1_downsampled = np.random.choice(w_class1, size=n_class0, replace=False)

print(); print(np.hstack((y[w_class0], y[w_class1_downsampled])))


# In[57]:


detailVol ='C:\\Users\\Nivas\\Downloads\\detailVol.csv'
df2=pd.read_csv(detailVol)
df2.head(10)


# In[60]:


df2.Step_Name = pd.factorize(df1.Step_Name)[0]


# In[61]:


X = df2.iloc[:,0:6]
y = df2['Step_Name']

y = np.where((y == 0), 0, 1)
print("Viewing the imbalanced target vector:\n", y)


# In[62]:


w_class0 = np.where(y == 0)[0]
w_class1 = np.where(y == 1)[0]

n_class0 = len(w_class0) 
n_class1 = len(w_class1)
 

print("n_class0: ", n_class0)
print("n_class1: ", n_class1)


# In[63]:


w_class1_downsampled = np.random.choice(w_class1, size=n_class0, replace=False)

print(); print(np.hstack((y[w_class0], y[w_class1_downsampled])))


# Apply low pass filter technique for noise removal on the data set for 'detailVol.csv' 

# In[64]:


# import required library
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


# In[72]:


# Specifications of the filter
f1 = 25  # Frequency of 1st signal
f2 = 50  # Frequency of 2nd signal
N = 10  # Order of the filter
fs=1000
  
# Generate the time vector of 1 sec duration
t = np.linspace(0, 1, 1000)  # Generate 1000 samples in 1 sec
  
# Generate the signal containing f1 and f2
sig = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)


# In[73]:


# Display the signal
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, sig)
ax1.set_title('25 Hz and 50 Hz sinusoids')
ax1.axis([0, 1, -2, 2])
  


# In[ ]:


# Design the Butterworth filter using signal.butter and output='sos'
# START CODE HERE ### (≈ 1 line of code)
sos = signal.butter(50, 35, 'lp', fs=1000, output='sos')

# Filter the signal by the filter using signal.sosfilt
# START CODE HERE ### (≈ 1 line of code)
# Use signal.sosfiltfilt to get output inphase with input
#filtered = signal.sosfiltfilt(sos, sig)
  
  
# Display the output signal
ax2.plot(t, filtered)
ax2.set_title('After 35 Hz Low-pass filter')
ax2.axis([0, 1, -2, 2])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()

