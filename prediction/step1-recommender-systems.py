import numpy as np
import pandas as pd
import os.path
import random
from random import randint
from random import uniform

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
                                  dtype={'userID': 'int', 'movieID': 'int', 'rating': 'int'},
                                  names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)


#####
##
## COLLABORATIVE FILTERING
##
#####

def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator


def similarity_matrix_with_cosine(ratingMatrix):
    number_items = np.shape(ratingMatrix)[0]
    similarity_matrix = np.zeros((number_items, number_items), dtype=object)

    for i in range(number_items):
        print(i)
        for j in range(i, number_items):
            movieID = j + 1
            similarity_matrix[i][j] = (movieID, cosine_similarity(ratingMatrix[:, i], ratingMatrix[:, j]))
            similarity_matrix[j][i] = similarity_matrix[i][j]

    return similarity_matrix


def predict_collaborative_filtering(movies, users, ratings, predictions):
    # Processing predictions data in order to return it from this function
    number_predictions = len(predictions)
    prediction_creating = [[idx, random.uniform(0, 5)] for idx in range(1, number_predictions + 1)]
    predictions_ratings = pd.DataFrame(prediction_creating, columns=['Id', 'Rating'])
    predictions_ratings['movieID'] = predictions['movieID']
    predictions_ratings['userID'] = predictions['userID']

    # Adding missing movie_ids to the numpy arrays
    range_missing = range(3696, 3707)

    # Creating utility matrix 'u' : User x Movie -> Rating
    utility_matrix = ratings.pivot_table(index='movieID', columns='userID', values='rating', fill_value=0)

    original_rating = utility_matrix.values
    for i, row in utility_matrix.iterrows():
        if i in range_missing:
            original_rating = np.vstack([original_rating, row.values])

    # Creating matrix for cosine similarity
    r = ratings.groupby('movieID', as_index=False, sort=False).mean().rename(columns={'movieID': 'movieID', 'rating': 'mean_rating'})
    r.drop('userID', axis=1, inplace=True)

    new_r = ratings.merge(r, how='left', on='movieID', sort=False)
    new_r['centered_cosine'] = new_r['rating'] - new_r['mean_rating']

    centered_cosine = new_r.pivot_table(index='movieID', columns='userID', values='centered_cosine', fill_value=0)

    all_movies_numpy = centered_cosine.values
    for i, row in centered_cosine.iterrows():
        if i in range_missing:
            all_movies_numpy = np.vstack([all_movies_numpy, row.values])

    ##########################
    #                        #
    # ALGORITHM STARTS HERE  #
    #                        #
    ##########################

    # Similarity matrix
    similarity_matrix = similarity_matrix_with_cosine(all_movies_numpy)

    # Mean of the ratings
    mean_all_ratings = ratings['rating'].mean()

    # Predicting ratings
    for i, user_movie in predictions.iterrows():
        # current_rating = original_rating[user_movie['movieID'] - 1][user_movie['userID'] - 1]
        # if current_rating > 0:
        #     predictions_ratings.at[i, 'Rating'] = current_rating
        #     continue

        # Calculating global effects
        user_calculate_mean = original_rating[:, user_movie['userID'] - 1]
        movie_calculate_mean = original_rating[user_movie['movieID'] - 1, :]

        mean_user_rating = user_calculate_mean[np.nonzero(user_calculate_mean)].mean()
        mean_movie_rating = movie_calculate_mean[np.nonzero(movie_calculate_mean)].mean()

        b_x = mean_user_rating - mean_all_ratings
        b_i = mean_movie_rating - mean_all_ratings

        # Get N similar items
        top_N_similar_movies = sorted(similarity_matrix[user_movie['movieID'] - 1], key=lambda pair: pair[1],
                                      reverse=True)
        similar_movies = top_N_similar_movies[1:4]

        # Predicting the rating with Pearson correlation
        pearson_denominator = 0
        for i, pair in enumerate(similar_movies):
            pearson_denominator += similar_movies[i][1]

        pearson_numerator = 0
        for i in range(0, 3):
            calculate_similar_movie = original_rating[similar_movies[i][0] - 1, :]
            mean_similar_movie = calculate_similar_movie[np.nonzero(calculate_similar_movie)].mean()

            b_xj = mean_similar_movie - mean_all_ratings
            pearson_numerator += similar_movies[i][1] * (original_rating[similar_movies[i][0] - 1]
                                                         [user_movie['userID'] - 1] - b_xj)

        # Modelling global and local effects with weighted average
        final_prediction = mean_all_ratings + b_x + b_i + (pearson_numerator / pearson_denominator)
        if final_prediction < 1:
            predictions_ratings.at[i, 'Rating'] = 1
        elif final_prediction > 5:
            predictions_ratings.at[i, 'Rating'] = 5
        else:
            predictions_ratings.at[i, 'Rating'] = final_prediction

    return predictions_ratings


#####
##
## LATENT FACTORS
##
#####

def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    # Processing predictions data in order to return it from this function
    number_predictions = len(predictions)
    prediction_creating = [[idx, random.uniform(0, 5)] for idx in range(1, number_predictions + 1)]
    predictions_ratings = pd.DataFrame(prediction_creating, columns=['Id', 'Rating'])
    predictions_ratings['movieID'] = predictions['movieID']
    predictions_ratings['userID'] = predictions['userID']

    # Adding missing movie_ids to the numpy arrays
    range_missing = range(3696, 3707)

    # Creating utility matrix 'u' : User x Movie -> Rating
    utility_matrix = ratings.pivot_table(index='movieID', columns='userID', values='rating', fill_value=0)

    original_rating = utility_matrix.values
    for i, row in utility_matrix.iterrows():
        if i in range_missing:
            original_rating = np.vstack([original_rating, row.values])

    # Creating matrix for cosine similarity
    r = ratings.groupby('movieID', as_index=False, sort=False).mean().rename(columns={'movieID': 'movieID', 'rating': 'mean_rating'})
    r.drop('userID', axis=1, inplace=True)

    new_r = ratings.merge(r, how='left', on='movieID', sort=False)
    new_r['centered_cosine'] = new_r['rating'] - new_r['mean_rating']

    centered_cosine = new_r.pivot_table(index='movieID', columns='userID', values='centered_cosine', fill_value=0)

    all_movies_numpy = centered_cosine.values
    for i, row in centered_cosine.iterrows():
        if i in range_missing:
            all_movies_numpy = np.vstack([all_movies_numpy, row.values])

    ##########################
    #                        #
    # ALGORITHM STARTS HERE  #
    #                        #
    ##########################

    # Doing Matrix factorization Q * PT
    U, S, VT = np.linalg.svd(original_rating, full_matrices=False)

    Q = U
    S_diagonal = np.diag(S)
    P = S_diagonal.dot(VT)

    # Predicting rating
    for i, user_movie in predictions.iterrows():
        qi = Q[user_movie['movieID'] - 1, :]
        px = P[:, user_movie['userID'] - 1]

        print(qi.dot(px))
        predictions_ratings.at[i, 'Rating'] = qi.dot(px)

    return predictions_ratings


#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
    ## TO COMPLETE
    # Processing predictions data in order to return it from this function
    number_predictions = len(predictions)
    prediction_creating = [[idx, random.uniform(0, 5)] for idx in range(1, number_predictions + 1)]
    predictions_ratings = pd.DataFrame(prediction_creating, columns=['Id', 'Rating'])
    predictions_ratings['movieID'] = predictions['movieID']
    predictions_ratings['userID'] = predictions['userID']

    '''
    Splitting known ratings into training and test data
    '''
    split_data = np.random.rand(len(ratings)) < 0.7
    train_data = ratings[split_data]
    test_data = ratings[~split_data]

    pass


rating_predictions = predict_collaborative_filtering(movies_description,
                                                     users_description, ratings_description, predictions_description)
print(rating_predictions.head())


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
