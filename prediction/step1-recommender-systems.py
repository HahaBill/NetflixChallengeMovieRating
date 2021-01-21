import numpy as np
import pandas as pd

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

def predict_collaborative_filtering_user_user(movies, users, ratings, predictions, neighbours, min_periods=27,
                                              print_output=False):
    predictions_ratings = []

    utility_matrix_none = ratings.pivot_table(index='userID', columns='movieID', values='rating',
                                              fill_value=None)

    # Add columns to the utility matrix for movies that are never rated
    cols = utility_matrix_none.columns
    for i in movies['movieID'].values:
        if i not in cols:
            utility_matrix_none[i] = np.nan

    utility_matrix_none = utility_matrix_none.transpose()
    cols = utility_matrix_none.columns
    for i in users['userID'].values:
        if i not in cols:
            utility_matrix_none[i] = np.nan

    utility_matrix_none.to_csv('util.csv')

    corr = utility_matrix_none.corr(min_periods=min_periods)

    # For every prediction to make (user/user in this case)
    for i in range(len(predictions)):
        if i % 100 == 0:
            print(i, "/", len(predictions))
        user = predictions.iloc[i][0]
        movie = predictions.iloc[i][1]

        c = corr[['userID', str(user)]]

        # Sort the pearson correlation for all movies to the current movie to predict
        sorted_pearson = c.sort_values(by=[str(user)], axis=0, ascending=False)

        # Delete the movie itself, it should not be checked
        sorted_pearson = sorted_pearson[sorted_pearson.userID != user]

        # Get the movie id's of the sorted movies
        sorted_users = sorted_pearson['userID'].values
        sorted_corr = sorted_pearson[str(user)].values

        # Add a certain amount of nearest neighbours, this amount is specified by the n_neighbours variable
        relevant_ratings = []
        for u in range(0, len(sorted_users)):
            us = sorted_users[u]
            rating = utility_matrix_none.at[movie, us]
            if not np.isnan(rating):
                relevant_ratings.append((rating, sorted_corr[u]))
                if len(relevant_ratings) == neighbours:
                    break

        relevant_ratings = np.array(relevant_ratings)
        pred = -1
        if len(relevant_ratings) > 0:
            total_weight = np.sum(relevant_ratings, axis=0)[-1]
            for j in range(len(relevant_ratings)):
                pred += relevant_ratings[j, 0] * relevant_ratings[j, 1] / total_weight

        # If the rating can't be calculated, set it to 3 as average
        if np.isnan(pred) or pred == -1:
            pred = 3

        if print_output:
            print("\n>>>>>>>>>>>>STARTING PREDICTION NUMBER", i + 1, "\nUser:", user, "\nMovie:", movie, "\n")
            print("\n>>SORTED PEARSON CORRELATION MATRIX\n")
            print(sorted_pearson)
            print("\n>>RELEVANT RATINGS AND THEIR WEIGHTS\n")
            print(relevant_ratings)
            print("\n>>FINAL PREDICTION: ", pred)
        predictions_ratings.append((i + 1, pred))
    return predictions_ratings


# Predict the submission using Collaborative filtering and put it in csv file (submission_collaborative_filtering.csv)
min_elements_non_zero = 27
n_neighbours = 30

preds_collaborative = predict_collaborative_filtering_user_user(movies_description,
                                                                users_description, ratings_description,
                                                                predictions_description,
                                                                n_neighbours, min_elements_non_zero)

predictions_cf = pd.DataFrame(preds_collaborative, columns=['Id', 'Rating'])
predictions_cf.to_csv('submission_collaborative_filtering.csv', index=False)


# ################################################################################################################################

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

        # Calculating user average rating
        user_rating = utility_matrix_none.loc[user].values
        mean_user_rating = np.nanmean(user_rating)

        pred = mean_user_rating + np.dot(qi, px)

        if np.isnan(pred) or pred < 1:
            pred = mean_all_ratings

        predictions_ratings.append((i + 1, pred))

    return predictions_ratings


# Predict the submission using Latent factors and put it in csv file (submission_latent_factors.csv)
preds_latent_factors = predict_latent_factors(movies_description, users_description, ratings_description,
                                              predictions_description)
predictions_latent_factors = pd.DataFrame(preds_latent_factors, columns=['Id', 'Rating'])
predictions_latent_factors.to_csv('submission_latent_factors.csv', index=False)
