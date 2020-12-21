import numpy as np
import pandas as pd
import os.path
from random import randint

print(np.version.version)
# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

# Where data is located
movies_file = '../data/movies.csv'
users_file = '../data/users.csv'
ratings_file = '../data/ratings.csv'
predictions_file = '../data/predictions.csv'
submission_file = '../data/submission.csv'

# movies_file = r'/prediction/data/movies.csv'
# users_file = '/prediction/data/users.csv'
# ratings_file = '/prediction/data/ratings.csv'
# predictions_file = '/prediction/data/predictions.csv'
# submission_file = '/data/submission.csv'

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID': 'int', 'year': 'int', 'movie': 'str'},
                                 names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';',
                                dtype={'userID': 'int', 'gender': 'str', 'age': 'int', 'profession': 'int'},
                                names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';',
                                  dtype={'userID': 'int', 'movieID': 'int', 'rating': 'int'},
                                  names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)


#####
##
## COLLABORATIVE FILTERING
##
#####

def predict_collaborative_filtering(movies, users, ratings, predictions):
    """
    Trying nearest-neighbor algorithm with Item-Item approach
    """
    # Creating utility matrix 'u' : User x Movie -> Rating
    utility_matrix = ratings \
        .pivot(index='movieID', columns='userID', values='rating') \
        .fillna(0)

    # Creating matrix for cosine similarity
    r = ratings \
        .groupby('movieID', as_index=False, sort=False) \
        .mean() \
        .rename(columns={'movieID': 'movieID', 'rating': 'mean_rating'})
    r.drop('userID', axis=1, inplace=True)

    new_r = ratings.merge(r, how='left', on='movieID', sort=False)
    new_r['centered_cosine'] = new_r['rating'] - new_r['mean_rating']

    centered_cosine = new_r \
        .pivot_table(index='movieID', columns='userID', values='centered_cosine') \
        .fillna(0)

    # Cosine similarity - find similar users for a certain user
    # Will finish later like tomorrow
#     print(new_r.values)


pass


#####
##
## LATENT FACTORS
##
#####

def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


predict_collaborative_filtering(movies_description, users_description, ratings_description, predictions_description)


#####
##
## RANDOM PREDICTORS
## //!!\\ TO CHANGE
##
#####

# By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


#####
##
## SAVE RESULTS
##
#####    


# ## //!!\\ TO CHANGE by your prediction function

'''
Carefully read this : You can uncomment this but I am doing my own submission output since I cannot
do "with open(submission_file)
- written by Bill
'''
# submission_read = pd.read_csv(submission_file)
# submission_read.columns = ['id', 'rating']
#
# predictions = predict_random(movies_description, users_description, ratings_description, predictions_description)
# predictions_df = pd.DataFrame(predictions, columns = ['Id', 'Rating'])
#
# submission_result = submission_read.merge(predictions_df, how='left', left_on='id', right_on='Id')
# submission_result.drop('id', axis=1, inplace=True)
# submission_result.drop('rating', axis=1, inplace=True)
#
# submission_result.to_csv('submission.csv',  index=False)

#
# # Save predictions, should be in the form 'list of tuples' or 'list of lists'
# with open(submission_file, 'w') as submission_writer:
#     #Formates data
#     predictions = [map(str, row) for row in predictions]
#     predictions = [','.join(row) for row in predictions]
#     predictions = 'Id,Rating\n'+'\n'.join(predictions)
#
#     #Writes it down
#     submission_writer.write(predictions)
