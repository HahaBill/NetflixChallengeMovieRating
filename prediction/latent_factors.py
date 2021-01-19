import numpy as np
import pandas as pd
import os.path
import random
from random import randint
from random import uniform

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
## LATENT FACTORS
##
#####

def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    # Processing predictions data in order to return it from this function
    predictions_ratings = []

    utility_matrix_svd = ratings.pivot_table(index='movieID', columns='userID', values='rating',
                                             fill_value=None)

    utility_matrix_svd.fillna(0, inplace=True)
    rating_numpy = []
    for i in range(1, 3707):
        if not i in utility_matrix_svd.index:
            rating_numpy.append(np.zeros(6040))
            continue
        else:
            rating_numpy.append(utility_matrix_svd.loc[i].values)

    ##########################
    #                        #
    # ALGORITHM STARTS HERE  #
    #                        #
    ##########################

    # Doing Matrix factorization Q * PT
    U, S, VT = np.linalg.svd(rating_numpy, full_matrices=False)

    Q = U
    S_diagonal = np.diag(S)

    # Choosing latent factors

    initial_energy = 0
    initial_energy = sum([np.square(S_diagonal[i][i]) for i in range(len(S_diagonal))])

    k = 2906

    for i in range(0, k):
        Q = np.delete(Q, len(Q[0]) - 1, 1)
        VT = np.delete(VT, len(VT) - 1, 0)

        S_diagonal = np.delete(S_diagonal, len(S_diagonal[0]) - 1, 1)
        S_diagonal = np.delete(S_diagonal, len(S_diagonal) - 1, 0)

    print("Number of "r" : ", len(S_diagonal))
    current_energy = sum([np.square(S_diagonal[i][i]) for i in range(len(S_diagonal))])
    percentage = current_energy / (initial_energy / 100)

    # Creating P matrix
    P = S_diagonal.dot(VT)

    # Use this matrix for calculating user/movie biases
    utility_matrix_none = ratings.pivot_table(index='userID', columns='movieID', values='rating',
                                              fill_value=None)

    # Add columns to the utility matrix for movies that are never rated
    cols = utility_matrix_none.columns
    for i in movies['movieID'].values:
        if i not in cols:
            utility_matrix_none[i] = np.nan

    # Mean of the ratings
    mean_all_ratings = ratings['rating'].mean()
    utility_matrix_none.replace(0, np.nan, inplace=True)

    # Predicting rating
    for i, user_movie in predictions.iterrows():
        if i % 100 == 0:
            print(i, "/", len(predictions))

        user = predictions.iloc[i][0]
        movie = predictions.iloc[i][1]

        qi = Q[movie - 1, :]
        px = P[:, user - 1]

        # Calculating global effects

        user_rating = utility_matrix_none.loc[user].values
        movie_rating = utility_matrix_none[movie].values

        mean_user_rating = np.nanmean(user_rating)
        mean_movie_rating = np.nanmean(movie_rating)

        b_x = mean_user_rating - mean_all_ratings
        b_i = mean_movie_rating - mean_all_ratings

        baseline = mean_all_ratings + b_i + b_x

        pred = baseline + np.dot(qi, px)

        if np.isnan(pred) or pred < 1:
            pred = mean_all_ratings

        predictions_ratings.append((i + 1, pred))

    return predictions_ratings


preds_latent_factors = predict_latent_factors(movies_description, users_description, ratings_description,
                                              predictions_description)
predictions_latent_factors = pd.DataFrame(preds_latent_factors, columns=['Id', 'Rating'])
predictions_latent_factors.to_csv('submission_latent_factors.csv', index=False)
