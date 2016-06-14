"""
Created on Mon May 30 17:38:41 2016

@author: Alex Wigmore
"""

import os
import subprocess
import collections
import re
import csv
import json

import pandas as pd
import numpy as np
import scipy

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

import psycopg2
from sqlalchemy import create_engine
import requests
from imdbpie import Imdb
import nltk

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#########################################
# part 1
# importing top 250 movies from imdb database using api thang into a dataframe
imdb = Imdb()
imdb = Imdb(anonymize=True)
top_250 = pd.DataFrame(imdb.top_250())

# sorting values by rating and selecting only the top 100 movies
top_250 = top_250.sort_values(by='rating', ascending=False)
top_100 = top_250[0:100]

# limiting columns according to starter code
mask = ['num_votes', 'rating', 'tconst', 'title', 'year']
top_100 = top_100[mask]

# getting genre/runtime from OMDB
top_100
movie_list = top_100['tconst']

def get_genre_runtime(b):
    genres = []
    runtime = []
    for i in b:
        request = requests.get('http://www.omdbapi.com/?i='+i+'&plot=short&r=json')
        r = request.json()
        genres.append(r['Genre'])
        runtime.append(r['Runtime'])
    answer =  pd.DataFrame({'genre':genres, 'runtime':runtime, 'tconst':movie_list})
    return answer

df = get_genre_runtime(movie_list)

# formatting data
df['runtime'] = df['runtime'].apply(lambda x: x.strip(' min'))
df['runtime'] = pd.to_numeric(df['runtime'])

# removing multiple genre names
def after_stripper(a):
    head, sep, tail = a.partition(',')
    return head

df['genre'] = df['genre'].apply(lambda x: after_stripper(x))

# merging values
df.reset_index(drop=True)
top_100.reset_index(drop=True)

top_100_clean = pd.merge(top_100, df)
top_100_clean
top_100_clean.to_csv('top_100.csv', encoding = 'utf-8', index=False)

#########################################
# part 2 - reviews and such - unable to do scraping, geting reviews from the omdb api
# reading in product of above code from .csv for time and such

# formatting shit for reviews
movie_list = pd.read_csv('top_100.csv')
movie_list = movie_list['tconst']

review_list = []
for i in movie_list:
    x = imdb.get_title_reviews(i, max_results=10000)
    review_list.append(x)
    print i


master_reviews = []
counter = 0
for i in review_list:
    imdb_value = movie_list[counter]
    for j in i:
        text = j.text
        score = j.rating
        master_reviews.append({'imdb_value':imdb_value, 'text':text, 'user_movie_score':score})
    counter +=1
len(master_reviews)
master_reviews = pd.DataFrame(master_reviews)
master_reviews['text'] = master_reviews['text'].apply(lambda x: re.sub(r'\W+', ' ', x))
master_reviews.dropna(inplace=True)
master_reviews.to_csv('reviews.csv', encoding = 'utf-8', index=False)


# creating dense matrix for review text data with TfidfVectorizer and such

# reading in file from csv to not repeat the lengthy code above
reviews_final = pd.read_csv('reviews.csv')


vectorizer = TfidfVectorizer(ngram_range = (1, 2), stop_words = 'english',
                                binary = False, max_features = 200)


vectorizer.fit(reviews_final['text'])

dense_matrix = pd.DataFrame(vectorizer.transform(reviews_final['text']).todense(), columns=vectorizer.get_feature_names())
reviews_final = reviews_final.join(dense_matrix)
reviews_final.to_csv('reviews_final.csv', encoding = 'utf-8', index=False)

#########################################
# using stemmed words to see if it produces better predict score correlation
# stemming shit
tester = pd.read_csv('reviews_raw.csv')

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
# and http://stackoverflow.com/questions/26126442/combining-text-stemming-and-removal-of-punctuation-in-nltk-and-scikit-learn
# stemmer that iterates over all elements of a tokenized list
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

# incorporates above stemmer with a tokenizer function
def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

# instantiating vectorizer with above tokenizer/stemmer
vectorizer2 = TfidfVectorizer(tokenizer=tokenize, ngram_range = (1, 2), stop_words = 'english',
                                binary = False, max_features = 200)

vectorizer2.fit(tester['text'])
dense_matrix2 = pd.DataFrame(vectorizer2.transform(tester['text']).todense(), columns=vectorizer2.get_feature_names())
tester = tester.join(dense_matrix2)
tester.to_csv('reviews_stemmed.csv', encoding = 'utf-8', index=False)

#########################################
# Part 3 - the sequel... get it? second verse, same as the first (data frame)
# engine = create_engine('postgresql://canadasfinest189@localhost:5432/project_6')
# engine.connect()
# stemmed_reviews.to_sql('stemmed_reviews', engine, flavor='postgres', if_exists='replace')
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
stemmed_reviews = pd.read_csv('reviews_stemmed.csv')
reviews = pd.read_csv('reviews_final.csv')
movies = pd.read_csv('top_100.csv')

# merging reviews with top_100 data in 2 separate dataframes
stemmed_full = pd.merge(left=stemmed_reviews, right=movies, how='left', left_on='imdb_value',right_on='tconst')
reviews_full = pd.merge(left=reviews, right=movies, how='left', left_on='imdb_value',right_on='tconst')

# removing extra columns
stemmed_full.pop('tconst')
reviews_full.pop('tconst')

# Visualizations part:
# runtime/year
sns.regplot(movies['year'], movies['runtime'])

# barplot of genres:
labels = movies['genre'].unique()
data = movies['genre'].value_counts()
sns.barplot(x=labels, y=data)
plt.gcf().set_size_inches(10, 4)


# total votes/movies
sns.stripplot(data= movies['num_votes'], jitter=True, alpha=.5)

# wtf lawlz roflcopter annoyance as fuck grumble

pizza = pd.DataFrame(movies.groupby(['genre'])['runtime'].mean())
labels = pizza['genre']
data = pizza['runtime']
sns.barplot(x=labels, y=data)
plt.gcf().set_size_inches(10, 4)


# runtime/genre

# segmenting
x = stemmed_full.columns
x = x[3:202]

##########################################################################################################################
#########################################################################################################################
# Part 5: Decision Tree Classifiers and Regressors

# setting up dummy variables for genre
genre_dummies = pd.DataFrame(pd.get_dummies(stemmed_full['genre']))
genre_columns=genre_dummies.columns
stemmed_full = pd.concat([stemmed_full, genre_dummies], axis=1)


# decision tree shit - regressing to predict score values cross validating dat shit

x_mask = ['10', '2', 'act', 'action', 'actor', 'actual', 'alway', 'amaz',
       'american', 'ani', 'anoth', 'anyth', 'audienc', 'away', 'bad',
       'batman', 'beauti', 'becaus', 'becom', 'befor', 'begin', 'believ',
       'best', 'better', 'big', 'bit', 'book', 'brilliant', 'cast',
       'chang', 'charact', 'cinema', 'classic', 'come', 'complet', 'creat',
       'dark', 'dark knight', 'day', 'definit', 'did', 'didn', 'didn t',
       'differ', 'direct', 'director', 'doe', 'doesn', 'doesn t', 'don',
       'don t', 'dream', 'effect', 'emot', 'end', 'enjoy', 'especi',
       'everi', 'everyon', 'everyth', 'excel', 'expect', 'face', 'fact',
       'famili', 'fan', 'far', 'favorit', 'feel', 'fight', 'film', 'final',
       'follow', 'friend', 'good', 'got', 'great', 'greatest', 'guy', 'ha',
       'happen', 'hard', 'help', 'hi', 'histori', 'hope', 'hour', 'howev',
       'human', 'idea', 'job', 'joker', 'just', 'kill', 'knight', 'know',
       'leav', 'let', 'life', 'like', 'line', 'littl', 'live', 'll',
       'long', 'look', 'lot', 'love', 'm', 'make', 'man', 'mani',
       'masterpiec', 'mean', 'mind', 'minut', 'moment', 'movi', 'movi wa',
       'music', 'need', 'new', 'nolan', 'noth', 'old', 'onc', 'onli',
       'origin', 'oscar', 'peopl', 'perfect', 'perform', 'person',
       'pictur', 'place', 'play', 'plot', 'point', 'power', 'probabl',
       'quit', 'rate', 'read', 'real', 'realli', 'reason', 'review',
       'right', 'ring', 'role', 's', 'said', 'saw', 'say', 'scene',
       'screen', 'second', 'seen', 'sens', 'set', 'shot', 'simpli', 'sinc',
       'someth', 'special', 'star', 'start', 'stori', 'sure', 't', 'tell',
       'thi', 'thi film', 'thi movi', 'thing', 'think', 'thought', 'time',
       'tri', 'true', 'truli', 'turn', 'understand', 'use', 've', 'veri',
       'view', 'viewer', 'wa', 'want', 'war', 'watch', 'watch thi', 'way',
       'whi', 'wonder', 'word', 'work', 'world', 'runtime', 'year_y', 'Action',
       'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Film-Noir',
       'Horror', 'Mystery', 'Western']

x2_mask = ['10', '2', 'act', 'action', 'actor', 'actual', 'alway', 'amaz',
       'american', 'ani', 'anoth', 'anyth', 'audienc', 'away', 'bad',
       'batman', 'beauti', 'becaus', 'becom', 'befor', 'begin', 'believ',
       'best', 'better', 'big', 'bit', 'book', 'brilliant', 'cast',
       'chang', 'charact', 'cinema', 'classic', 'come', 'complet', 'creat',
       'dark', 'dark knight', 'day', 'definit', 'did', 'didn', 'didn t',
       'differ', 'direct', 'director', 'doe', 'doesn', 'doesn t', 'don',
       'don t', 'dream', 'effect', 'emot', 'end', 'enjoy', 'especi',
       'everi', 'everyon', 'everyth', 'excel', 'expect', 'face', 'fact',
       'famili', 'fan', 'far', 'favorit', 'feel', 'fight', 'film', 'final',
       'follow', 'friend', 'good', 'got', 'great', 'greatest', 'guy', 'ha',
       'happen', 'hard', 'help', 'hi', 'histori', 'hope', 'hour', 'howev',
       'human', 'idea', 'job', 'joker', 'just', 'kill', 'knight', 'know',
       'leav', 'let', 'life', 'like', 'line', 'littl', 'live', 'll',
       'long', 'look', 'lot', 'love', 'm', 'make', 'man', 'mani',
       'masterpiec', 'mean', 'mind', 'minut', 'moment', 'movi', 'movi wa',
       'music', 'need', 'new', 'nolan', 'noth', 'old', 'onc', 'onli',
       'origin', 'oscar', 'peopl', 'perfect', 'perform', 'person',
       'pictur', 'place', 'play', 'plot', 'point', 'power', 'probabl',
       'quit', 'rate', 'read', 'real', 'realli', 'reason', 'review',
       'right', 'ring', 'role', 's', 'said', 'saw', 'say', 'scene',
       'screen', 'second', 'seen', 'sens', 'set', 'shot', 'simpli', 'sinc',
       'someth', 'special', 'star', 'start', 'stori', 'sure', 't', 'tell',
       'thi', 'thi film', 'thi movi', 'thing', 'think', 'thought', 'time',
       'tri', 'true', 'truli', 'turn', 'understand', 'use', 've', 'veri',
       'view', 'viewer', 'wa', 'want', 'war', 'watch', 'watch thi', 'way',
       'whi', 'wonder', 'word', 'work', 'world', 'year_y', 'Action',
       'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Film-Noir',
       'Horror', 'Mystery', 'Western', 'user_movie_score']

len(x2_mask)
# segmenting variables
y = stemmed_full['user_movie_score']
X = stemmed_full[x_mask]

# instantiating regressor and cross validating stuff
# regressor doesn't work for predicting these...
trees = DecisionTreeRegressor()
trees.fit(X, y)
cross_val_score(trees, X, y, cv=5, n_jobs=-1)

# regression attempt to predict runtime
X2 = stemmed_full[x2_mask]
y2 = stemmed_full['runtime']
trees4 = DecisionTreeRegressor()
trees4.fit(X2, y2)
cross_val_score(trees4, X2, y2, cv=5, n_jobs=-1)

# Decision Tree Classifiying the score
trees2 = DecisionTreeClassifier()
trees2.fit(X, y)
cross_val_score(trees2, X, y, cv=5, n_jobs=-1)

# gridsearching the classifier
grid_classifier = {'max_depth':depth, 'max_features':x}
x = range(1,200,10)
depth = [1,2,3,4,5]
trees3 = DecisionTreeClassifier()
grid = GridSearchCV(trees2, grid_classifier)
grid.fit(X,y)
grid.best_score_
0.52544263209256936
grid.best_params_
{'max_depth': 1, 'max_features': 1}

X.columns.values
## creates a gosh darn node!
# best_params_ = max_depth:1, max_features:1... lawlz roflcopter



######################################################################################################################
# Part 7... Rando Forest and Such
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier

# train test split to get accuracy score for classification models
X_train, X_test, y_train, y_test = train_test_split(X, y)


# bagging model
bdt = BaggingClassifier()

def do_cross_val(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, n_jobs=-1, verbose=1)
    return scores.mean(), scores.std()

# cross val scores with a bagging/decision tree classifier
do_cross_val(bdt)
(0.38116005470351122, 0.052192121705746754)


# Random Forest Classification with crossval

forest = RandomForestClassifier()
do_cross_val(forest, X, y)
#random forrest classifier - (0.45703255495422573, 0.027952293163319378)
forest.n_features_
forest.fit(X,y)


# adaboost classifer with cross cal
ada = AdaBoostClassifier()

do_cross_val(ada, X, y)
(0.51081392284531635, 0.024275400766445469)
ada.fit(X,y)
pd.DataFrame(ada.predict_proba(X_test))

# Extra Trees
x_trees = ExtraTreesClassifier()
do_cross_val(x_trees, X, y)
# Extra trees (0.43864060302745989, 0.036794626953118421)

# Random Forest paramters & plotting and such
importance = pd.DataFrame({"Features": X.columns.values, 'model_importance (multiplied by 1000)':forest.feature_importances_})

importance['model_importance (multiplied by 1000)'] = importance['model_importance (multiplied by 1000)'].apply(lambda x: x*1000)
importance = importance.sort_values(by ='model_importance (multiplied by 1000)', ascending=False)

x_plot = range(len(importance['Features']))
plt.figure(figsize = (20, 15))
plt.xticks(x_plot, importance['Features'], rotation = 90)

plt.bar(x_plot, importance['model_importance (multiplied by 1000)'])
plt.show()
