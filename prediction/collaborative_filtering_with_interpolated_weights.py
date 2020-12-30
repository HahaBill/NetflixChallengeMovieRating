import numpy as np
import pandas as pd
from random import randint

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
## COLLABORATIVE FILTERING WITH WEIGHTED SUM
## - Implementation with interpolated weights
## - Gradient descent
##
#####

'''
Computing cosine similarity
'''


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator


'''
Creating similarity matrix with cosine similarity
'''


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


'''
Error sum of square
'''


def sum_of_squares(ratings_with_prediction):
    error_rating_col = ratings_with_prediction.apply(lambda row: np.square(row['predicted_rating'] - row['rating']),
                                                     axis=1)
    sse_error = error_rating_col.sum()

    return sse_error


'''
Interpolated weights + Gradient descent for Collaborative Filtering
'''


def interpolation_weights_optimization(movies, users, ratings, predictions, num_iterations, weight):
    w = weight
    i = 0
    error_threshold = 0.01
    current_error = 1000

    while i < num_iterations and current_error < error_threshold:
        result_predicted_ratings = predict_collaborative_filtering(movies, users, ratings, predictions, w)
        error_sse = sum_of_squares(result_predicted_ratings)

        current_error = error_sse
        i += 1

    return predictions


def predict_collaborative_filtering(movies, users, ratings, predictions, w):
    predictions['predicted_rating'] = np.random.randint(1, 6, predictions.shape[0])

    # Adding missing movie_ids to the numpy arrays
    range_missing = range(3696, 3707)

    # Creating utility matrix 'u' : User x Movie -> Rating
    utility_matrix = ratings.pivot_table(index='movieID', columns='userID', values='rating', fill_value=0)

    original_rating = utility_matrix.values
    for i, row in utility_matrix.iterrows():
        if i in range_missing:
            original_rating = np.vstack([original_rating, row.values])

    # Creating matrix for cosine similarity
    r = ratings.groupby('movieID', as_index=False, sort=False).mean().rename(
        columns={'movieID': 'movieID', 'rating': 'mean_rating'})
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

    # Average rating
    mean_all_ratings = ratings['rating'].mean()

    # Predicting ratings
    for i, user_movie in predictions.iterrows():

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

        # Predicting the rating with interpolated weights
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

        final_prediction = mean_all_ratings + b_x + b_i + (pearson_numerator / pearson_denominator)

        if final_prediction < 1:
            predictions.at[i, 'predicted_rating'] = 1
        elif final_prediction > 5:
            predictions.at[i, 'predicted_rating'] = 5
        else:
            predictions.at[i, 'predicted_rating'] = final_prediction

    return predictions


def predict(movies, users, ratings, predictions):
    # number_predictions = len(predictions)
    # return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]

    '''
    Splitting known ratings into training and test data
    '''
    split_data = np.random.rand(len(ratings)) < 0.7
    train_data = ratings[split_data]
    test_data = ratings[~split_data]

    ratings['predicted_rating'] = np.random.randint(1, 6, ratings.shape[0])

    sse = sum_of_squares(ratings)
