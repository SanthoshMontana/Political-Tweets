#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
nltk.download('stopwords')
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report 

stop_words = set(stopwords.words('english')) 
df = pd.read_csv('ExtractedTweets.csv')
                     
X = df['Tweet']
y = df['Party']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
clf = MultinomialNB().fit(X_train_counts, y_train)
X_new_counts = count_vect.transform(X_test)
predicted = clf.predict(X_new_counts)
print("Multinomial:  " + str(accuracy_score(y_test,predicted)))
cm = confusion_matrix(y_test, predicted)
df_show = pd.DataFrame()
df_show = pd.DataFrame(cm, index = ["Democrat","Republican"], columns = ["Democrat","Republican"])
print(df_show)
print(classification_report(y_test, predicted))

X_train_counts = count_vect.fit_transform(X_train)
clf = BernoulliNB().fit(X_train_counts, y_train)
X_new_counts = count_vect.transform(X_test)
predicted = clf.predict(X_new_counts)
print("Bernoulli:  " + str(accuracy_score(y_test,predicted)))
cm = confusion_matrix(y_test, predicted)
df_show = pd.DataFrame()
df_show = pd.DataFrame(cm, index = ["Democrat","Republican"], columns = ["Democrat","Republican"])
print(df_show)
print(classification_report(y_test, predicted))

