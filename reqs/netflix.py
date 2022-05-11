import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib


# ##  Predict how long a movie will last (duration)

# Before we proceed further, let's remove movies that have their duration in seasons, and TV shows with a duration in minutes.

#fill NAs

def preprocess(df):
    df['director'].fillna('unknown', inplace=True)
    df['cast'].fillna('unknown', inplace=True)
    df['country'].fillna('unknown', inplace = True)
    df['date_added'].fillna('00-00-00', inplace = True)
    df['release_year'].fillna(0, inplace = True)
    df['rating'].fillna('unknown', inplace = True)
    #remove movies that have their duration in seasons, and TV shows with a duration in minutes.
    df = df[~((df['type']=='Movie') & (df['duration'].str.contains('Season')))]
    df = df[~((df['type']=='TV Show') & (df['duration'].str.contains('min')))]
    df['duration_num'] = df['duration'].apply(lambda x: getDuration(x))
    df['yearAdded'] = df['date_added'].apply(lambda x: getYearAdded(x))
    df['release_year']=df['release_year'].astype('int')

    stop = stopwords.words('english') #remove stopwords
    df['new_desc'] = df['description'].apply(lambda s: s.lower())
    df['new_desc'] = df['new_desc'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    v = TfidfVectorizer(max_features=20)
    x = v.fit_transform(df['new_desc'])

    textData = pd.DataFrame(x.toarray(), columns = v.get_feature_names())

    df = df.reset_index() #or else concat doesn't use the right indices
    newdf = pd.concat([df,textData], axis=1)

    #get cast count
    newdf['cast_count'] = newdf['cast'].apply(lambda x: len(x.split(',')))
    #get dummies for country and rating
    newdf= pd.get_dummies(data=newdf, columns=['country', 'rating'])
    #one-hot encode the genres
    newdf['genres'] = newdf['listed_in'].apply(lambda x: x.split(','))

    genredf = pd.get_dummies(newdf['genres'].explode()).sum(level=0)
    newdf= pd.concat([newdf,genredf], axis=1)
    moviedf = newdf[newdf['type']=='Movie']
    return(moviedf)

def getDuration(duration):
    wordList = duration.split(' ')
    length = int(wordList[0])
    return length

def getYearAdded(dateAdded):
    if dateAdded == '00-00-00':
        return('0000')
    lastTwoDigYear = dateAdded[-2:]
    fourDigYear = int('20'+lastTwoDigYear)
    return(fourDigYear)

def getTrainTestXY(df):
    X_movie, y_movie = moviedf.loc[:, ~moviedf.columns.isin(['type','index','show_id', 'title', 'cast', 'index', 'director','date_added', 'duration', 'listed_in', 'description', 'genres', 'new_desc',  'yearAdded'])], moviedf['duration_num']
    #generate training and testing sets
    X_train_movie, X_test_movie, y_train_movie, y_test_movie = train_test_split(X_movie,y_movie, test_size = .2)
    return((X_train_movie, X_test_movie, y_train_movie, y_test_movie))

# defining parameter range 
param_grid = {'learning_rate': [0.1, 0.15, 0.2, 0.25],  
              'gamma': [0,1,10], 
              'max_depth': [5,7,9],
             'alpha':[0,5,10]}  

grid = GridSearchCV(xgb.XGBRegressor(), param_grid, refit = True, verbose = 1,n_jobs=-1) 
grid.fit(X_train_movie, y_train_movie) 


# In[73]:


print(grid.best_params_)


# So here we see that optimal parameters include `learning_rate = 0.1, max_depth=9,` and `alpha=gamma=0`.

# In[74]:


grid_predictions = grid.predict(X_test_movie)


# In[75]:


grid_predictions


# In[76]:


new_rmse = np.sqrt(mean_squared_error(y_test_movie, grid_predictions))
print("RMSE: %f" % (new_rmse))


# After using gridsearch to identify a better set of hyperparameter configurations, we have a new model now that has an root-mean-squared error of 0.2 minutes, substantially better than the orignal error of 12 minutes!

# In[77]:


joblib.dump(grid, 'netflix_xgboost.pkl')


# Now let's pickle this model and containerize it.
