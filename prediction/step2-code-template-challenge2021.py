import numpy as np
import pandas as pd
from random import randint

# -*- coding: utf-8 -*-
"""
FRAMEWORK FOR DATAMINING CLASS

#### IDENTIFICATION
NAME:
SURNAME:
STUDENT ID:
KAGGLE ID:


### NOTES
This files is an example of what your code should look like. 
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

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])


#####
##
## COLLABORATIVE FILTERING WITH BASELINE ESTIMATE
## - Implementation with item-based collaborative filtering
## - Incorporated global biases
##
#####
# Minimal elements to have a rating on for two movies to be considered a neighbour. Otherwise a movie with one rating
# and rest all zeroes is a good neighbour to all movies with that rating by that one user
def predict_collaborative_filtering(movies, users, ratings, predictions, neighbours, min_periods, print_output=False,
                                    corr=None):
    predictions_ratings = []

    # Creating utility matrix
    utility_matrix_none = ratings.pivot_table(index='userID', columns='movieID', values='rating',
                                              fill_value=None)

    # Add columns to the utility matrix for movies that are never rated
    cols = utility_matrix_none.columns
    for i in movies['movieID'].values:
        if i not in cols:
            utility_matrix_none[i] = np.nan

    if corr is None:
        corr = utility_matrix_none.corr(min_periods=min_periods)

    # Creating similarity matrix with pearson correlation, in order to not compute it multiple times, we
    # store the matrix in .csv file
    corr.to_csv(r'tempcorr.csv')
    corr = pd.read_csv(r'tempcorr.csv')

    # Average rating
    mean_all_ratings = ratings['rating'].mean()

    # For every prediction to make (item/item, or movie/movie in this case)
    for i in range(len(predictions)):
        if i % 100 == 0:
            print(i, "/", len(predictions))
        user = predictions.iloc[i][0]
        movie = predictions.iloc[i][1]

        # Calculating baseline estimates
        user_rating = utility_matrix_none.loc[user].values
        movie_rating = utility_matrix_none[movie].values

        mean_user_rating = np.nanmean(user_rating)
        mean_movie_rating = np.nanmean(movie_rating)

        b_x = mean_user_rating - mean_all_ratings
        b_i = mean_movie_rating - mean_all_ratings

        baseline = mean_all_ratings + b_i + b_x

        c = corr[['movieID', str(movie)]]

        # Sort the pearson correlation for all movies to the current movie to predict
        sorted_pearson = c.sort_values(by=[str(movie)], axis=0, ascending=False)

        # Delete the movie itself, it should not be checked
        sorted_pearson = sorted_pearson[sorted_pearson.movieID != movie]

        # Get the movie id's of the sorted movies
        sorted_movies = sorted_pearson['movieID'].values
        sorted_corr = sorted_pearson[str(movie)].values

        # Add a certain amount of nearest neighbours, this amount is specified by the n_neighbours variable
        relevant_ratings = []
        for m in range(0, len(sorted_movies)):
            mov = sorted_movies[m]
            rating = utility_matrix_none.at[user, mov]
            if not np.isnan(rating):
                relevant_ratings.append((rating, sorted_corr[m], mov))
                if len(relevant_ratings) == neighbours:
                    break

        relevant_ratings = np.array(relevant_ratings)

        # Predicting with weighted average
        total_weight = 0
        for x in relevant_ratings:
            total_weight = total_weight + abs(x[1])

        pred = 0
        for j in range(len(relevant_ratings)):
            current_movie = int(relevant_ratings[j, 2])
            calculate_similar_movie = utility_matrix_none[current_movie].values
            mean_similar_movie = np.nanmean(calculate_similar_movie)

            # Calculating baseline estimates for the current similar movie
            b_j = (mean_similar_movie - mean_all_ratings)

            b_xj = mean_all_ratings + b_j + b_x

            pred += (relevant_ratings[j, 0] - b_xj) * relevant_ratings[j, 1] / total_weight

        pred = baseline + pred

        # If the rating can't be calculated, set it to 3 as average
        if np.isnan(pred) or pred == 0:
            pred = mean_all_ratings

        if pred > 5:
            pred = 5

        if pred < 1:
            pred = 1

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

preds_collaborative = predict_collaborative_filtering(movies_description,
                                                      users_description, ratings_description, predictions_description,
                                                      n_neighbours, min_elements_non_zero)

predictions_cf = pd.DataFrame(preds_collaborative, columns=['Id', 'Rating'])
predictions_cf.to_csv('submission_collaborative_filtering.csv', index=False)


# ################################################################################################################################


#####
##
## LATENT FACTORS WITH BASELINE ESTIMATE
## - Incorporated global biases
## - Reduced latent features
##
#####
def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    # Processing predictions data in order to return it from this function
    predictions_ratings = []

    # Creating Utility matrix
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

    # Choosing latent features (optimal number of rank of the S matrix)
    initial_energy = 0
    initial_energy = sum([np.square(S_diagonal[i][i]) for i in range(len(S_diagonal))])

    # Latent features = 3706 - k
    k = 1806
    for i in range(0, k):
        Q = np.delete(Q, len(Q[0]) - 1, 1)
        VT = np.delete(VT, len(VT) - 1, 0)

        S_diagonal = np.delete(S_diagonal, len(S_diagonal[0]) - 1, 1)
        S_diagonal = np.delete(S_diagonal, len(S_diagonal) - 1, 0)

    # Calculating energy and showing the current latent features, using 80% - 90% rule of thumb
    num_latent_features = len(S_diagonal)
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


# Predict the submission using Latent factors and put it in csv file (submission_latent_factors.csv)
preds_latent_factors = predict_latent_factors(movies_description, users_description, ratings_description,
                                              predictions_description)
predictions_latent_factors = pd.DataFrame(preds_latent_factors, columns=['Id', 'Rating'])
predictions_latent_factors.to_csv('submission_latent_factors.csv', index=False)
