#!/usr/bin/env python
# coding: utf-8

# # Let's examine some data from Netflix

# In[56]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error


# In[2]:


df= pd.read_csv('new_netflix2.csv')


# In[3]:


df.head()


# In[4]:


#how many missing values?
df.isna().sum()


# In[5]:


#fill NAs
df['director'].fillna('unknown', inplace=True)
df['cast'].fillna('unknown', inplace=True)
df['country'].fillna('unknown', inplace = True)
df['date_added'].fillna('00-00-00', inplace = True)
df['release_year'].fillna(0, inplace = True)
df['rating'].fillna('unknown', inplace = True)


# Let's examine the duration of movies and tv shows separately:

# In[6]:


#unique types of content:
df.type.unique()


# In[7]:


tv_shows = df[df['type']=='TV Show']


# Now, we'd expect most TV shows to be have their durations listed in seasons, but perhaps there are some that don't?

# In[8]:


tv_shows[~tv_shows['duration'].str.contains('Season')]


# Interesting! There are some TV shows that have a duration listed in minutes. Let's filter these out.

# In[9]:


#now, extract numerical values
tv_shows = tv_shows[tv_shows['duration'].str.contains('Season')]
tv_shows['duration_num'] = tv_shows['duration'].apply(lambda x: int(x[0]))


# In[10]:


plt.hist(tv_shows.duration_num)
plt.xlabel('Seasons')
plt.ylabel('Number of Shows')
plt.show()


# So here we see that most TV shows run for 1 or 2 seasons, and very few run for more than 4 seasons. Neat!

# In[11]:


movies = df[df['type']=='Movie']


# Similarly, is it possible for a movie to have a duration in seasons?

# In[12]:


movies[movies['duration'].str.contains('Season')]


# In[13]:


movies = movies[movies['duration'].str.contains('min')]
movies['duration_num'] = movies['duration'].apply(lambda x: int(x[:-4])) # this should  remove ' min' from durations
plt.hist(movies.duration_num)
plt.xlabel('Duration in Minutes')
plt.ylabel('Number of Movies')
plt.show()


# After removing the movie with a 1 season duration, we see that most movies last around 100 minutes.

# Let's take a look at the ratings now:

# In[14]:


df['rating'].unique()


# In[15]:


df['rating'].value_counts().plot(kind='bar')


# Interesting! So we see that most content on Netflix has a TV-MA rating, followed by TV-14 and TV-PG.

# ##  Predict how long a movie or TV show will last (duration)

# Before we proceed further, let's remove movies that have their duration in seasons, and TV shows with a duration in minutes.

# In[16]:


df = df[~((df['type']=='Movie') & (df['duration'].str.contains('Season')))]
df = df[~((df['type']=='TV Show') & (df['duration'].str.contains('min')))]


# In[17]:


def getDuration(duration):

    wordList = duration.split(' ')
    length = int(wordList[0])
    return length


    
df['duration_num'] = df['duration'].apply(lambda x: getDuration(x))


# In[18]:


def getYearAdded(dateAdded):
    if dateAdded == '00-00-00':
        return('0000')
    lastTwoDigYear = dateAdded[-2:]
    fourDigYear = int('20'+lastTwoDigYear)
    return(fourDigYear)

df['yearAdded'] = df['date_added'].apply(lambda x: getYearAdded(x))


# In[19]:


df['release_year']=df['release_year'].astype('int')


# Now let's get common topics in the `description` column using TFIDF

# In[20]:


stop = stopwords.words('english') #remove stopwords
df['new_desc'] = df['description'].apply(lambda s: s.lower())
df['new_desc'] = df['new_desc'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
v = TfidfVectorizer(max_features=20)
x = v.fit_transform(df['new_desc'])


# In[21]:


textData = pd.DataFrame(x.toarray(), columns = v.get_feature_names())
textData.head()


# In[22]:


df = df.reset_index() #or else concat doesn't use the right indices
newdf = pd.concat([df,textData], axis=1)


# In[23]:


newdf['cast_count'] = newdf['cast'].apply(lambda x: len(x.split(',')))
newdf= pd.get_dummies(data=newdf, columns=['country', 'rating'])
#one-hot encode the genres
newdf['genres'] = newdf['listed_in'].apply(lambda x: x.split(','))
genredf = pd.get_dummies(newdf['genres'].explode()).sum(level=0)


# In[29]:


newdf= pd.concat([newdf,genredf], axis=1)


# Now, generate X and y matrices

# In[43]:


moviedf = newdf[newdf['type']=='Movie']
X_movie, y_movie = moviedf.loc[:, ~moviedf.columns.isin(['type','index','show_id', 'title', 'cast', 'index', 'director','date_added', 'duration', 'listed_in', 'description', 'genres', 'new_desc',  'yearAdded'])], moviedf['duration_num']


# In[44]:


X_movie


# In[46]:


data_dmatrix_movie = xgb.DMatrix(data=X_movie,label=y_movie ,enable_categorical = True)
X_train_movie, X_test_movie, y_train_movie, y_test_movie = train_test_split(X_movie,y_movie, test_size = .2)


# In[70]:


xg_reg_movie = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.9, learning_rate = 0.2,
                max_depth = 8, alpha = 12, n_estimators = 10)


# In[71]:


xg_reg_movie.fit(X_train_movie,y_train_movie)


# In[72]:


preds = xg_reg_movie.predict(X_test_movie)


# In[66]:


preds


# In[73]:


rmse = np.sqrt(mean_squared_error(y_test_movie, preds))
print("RMSE: %f" % (rmse))


# Wow! Our gradient boosting regression model has a root-mean-square error of about 12 minutes.
# Can we do better?
