#!/usr/bin/env python
# coding: utf-8

# # load the dataset

# In[2]:


import graphlab


# In[3]:


dataset = graphlab.SFrame.read_csv('C:\\Users\\Vidya sagar\\Documents\\ml-20m\\ml-20m1\\ratings.csv')


# In[5]:


dataset.head()


# In[7]:


items = graphlab.SFrame.read_csv('C:\\Users\\Vidya sagar\\Documents\\ml-20m\\ml-20m1\\movies.csv')


# In[8]:


items.head()


# In[9]:



items


# In[10]:


items['title'].sketch_summary()


# In[11]:


items.unique()


# In[12]:


items['movieId'].show()


# In[13]:


items['title'].show()


# In[14]:


items['movieId=53835'].show()


# In[21]:


mostly_liked_movie = items[items['movieId']==53835]


# In[23]:


mostly_liked_movie.show()


# In[25]:


training_data,test_data = dataset.random_split(.8,seed=0)


# In[28]:


model=graphlab.recommender.create(training_data,'userId','movieId')


# In[29]:


results = model.recommend()


# In[ ]:




