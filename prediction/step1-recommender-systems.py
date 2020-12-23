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
movies_description = pd.read_csv(movies_file, delimiter=';',
                                 dtype={'movieID': 'int', 'year': 'int', 'movie': 'str'},
                                 names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';',
                                dtype={'userID': 'int', 'gender': 'str', 'age': 'int', 'profession': 'int'},
                                names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';',
                                  dtype={'userID': 'int', 'movieID': 'int', 'rating': 'float64'},
                                  names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';',
                                      dtype={'userID': 'int', 'movieID': 'int'}, names=['userID', 'movieID'],
                                      header=None)


#####
##
## COLLABORATIVE FILTERING
##
#####


def predict_collaborative_filtering(movies, users, ratings, predictions):

    # Processing predictions data in order to return it from this function
    number_predictions = len(predictions)
    prediction_creating = [[idx, random.uniform(0, 5)] for idx in range(1, number_predictions + 1)]
    predictions_ratings = pd.DataFrame(prediction_creating, columns=['Id', 'Rating'])
    predictions_ratings['movieID'] = predictions['movieID']
    predictions_ratings['userID'] = predictions['userID']

    # Adding missing movie_ids to the numpy arrays
    range_missing = range(3696, 3707)

    '''
    Creating utility matrix 'u' : User x Movie -> Rating
    '''
    utility_matrix = ratings.pivot_table(index='movieID', columns='userID', values='rating',
                                         fill_value=0)

    original_rating = utility_matrix.values
    for i, row in utility_matrix.iterrows():
        if (i in range_missing):
            original_rating = np.vstack([original_rating, row.values])

    '''
    Creating matrix for cosine similarity
    '''
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

    all_movies_numpy = centered_cosine.values
    for i, row in centered_cosine.iterrows():
        if (i in range_missing):
            all_movies_numpy = np.vstack([all_movies_numpy, row.values])

    '''
    Cosine similarity - find similar users for a certain user based on |N|,
    also making a prediction with Pearson correlation
    '''
    for i, user_movie in predictions.iterrows():
        print("CURRENT MOVIE : ", user_movie['movieID'])
        current_movie = all_movies_numpy[user_movie['movieID'] - 1]
        current_rating = original_rating[user_movie['movieID'] - 1][user_movie['userID'] - 1]
        if (current_rating > 0):
            predictions_ratings.at[i, 'Rating'] = current_rating
            continue

        current_denominator = np.sqrt(sum([np.square(x) for x in current_movie]))
        top_N_similar_movies = []

        # Computing similarities to current movie that we want to predict for particular user
        for id_movie, movie in enumerate(all_movies_numpy):
            numerator = [x * y for x, y in zip(current_movie, movie)]
            other_denominator = np.sqrt(sum([np.square(x) for x in movie]))
            costheta = sum(numerator) / (current_denominator * other_denominator)
            top_N_similar_movies.append((id_movie + 1, costheta))

        # Get N similar items
        top_N_similar_movies.sort(key=lambda pair: pair[1], reverse=True)
        similar_movies = top_N_similar_movies[0:5]
        print("PAIR : ", "first element =", similar_movies[0][0], "second element =", similar_movies[0][1])

        # Predicting the rating with Pearson correlation
        pearson_denominator = sum([pair[1] for pair in similar_movies])
        pearson_numerator = 0
        for i in range(0, 5):
            pearson_numerator += similar_movies[i][1] * original_rating[similar_movies[i][0] - 1][
                user_movie['userID'] - 1]

        print("Predicting...", pearson_numerator, " / ", pearson_denominator)
        predictions_ratings.at[i, 'Rating'] = (pearson_numerator / pearson_denominator)
        print("Predicted rating : ", predictions_ratings.at[i, 'Rating'])

    return predictions_ratings

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


rating_predictions = predict_collaborative_filtering(movies_description,
                                                     users_description, ratings_description, predictions_description)


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
