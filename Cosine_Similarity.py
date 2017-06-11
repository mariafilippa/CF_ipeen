from time import time
from math import sqrt
import argparse

import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn import metrics

#根據與 user_id最相似的 k個 users推測評分矩陣
def topk_prediction(ratings, similarity_matrix, user_id, k):
    #前 k個最相似的 users
    indexes_k = np.argsort(similarity_matrix[user_id])[-2:-2-k:-1]
    #和 k users的相似度
    similarity_k = similarity_matrix[user_id][indexes_k]

    #計算 user_id的全部 ratings
    movie_to_user = np.array(ratings[indexes_k]).T
    similarity_dot_ratings = np.dot(movie_to_user, similarity_k)
    similarity_sum = np.dot((movie_to_user!=0.0).astype(int), similarity_k)
    ratings_prediction = similarity_dot_ratings / similarity_sum
    #移除 nan
    nan_removed = np.nan_to_num(ratings_prediction)
    return nan_removed


#統整全部 users的 ratings
def ratings_prediction(train_data_matrix, k):
    prediction_matrix = np.zeros_like(train_data_matrix)

    #只計算訓練集cosine similarity
    train_similarity = metrics.pairwise.cosine_similarity(train_data_matrix, train_data_matrix)
    #計算每個 user的 ratings
    for user in range(0, train_data_matrix.shape[0]-1, 1):
        prediction_matrix[user] = topk_prediction(train_data_matrix, train_similarity, user, k)

    return prediction_matrix


#計算誤差
def rmse(ground_truth, prediction):
    #只取測試集不為0的值來計算
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(metrics.mean_squared_error(ground_truth, prediction))


def main(df, k, user_id):
    n_users = df.user_reindex.unique().shape[0]
    n_restaurant = df.restaurant_reindex.unique().shape[0]
    print('%d users' % (n_users))
    print('%d items' % (n_restaurant))

    #建立訓練集和測試集
    train_data, test_data = ms.train_test_split(df_reindex, test_size=0.4)

    #建立評分矩陣
    train_data_matrix = np.zeros((n_users, n_restaurant))
    for line in train_data.itertuples():
        train_data_matrix[line[5], line[4]] = line[3]/10.0

    test_data_matrix = np.zeros_like(train_data_matrix)
    for line in test_data.itertuples():
        test_data_matrix[line[5], line[4]] = line[3]/10.0

    #計算預測評分
    ts = time()
    prediction_matrix = ratings_prediction(train_data_matrix, k)
    print(str(k) + " users : " + str(time() - ts) + "s")

    #計算誤差
    print("rmse : " + str(rmse(test_data_matrix, prediction_matrix)))

    #列出預測評分最高的10個 id
    if user_id >= 0:
        print("top recommendation : " + str(np.argsort(prediction_matrix[user_id])[-1:(-1-10):-1]))


if __name__ == '__main__':
    #讓使用者能改變參數
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--filepath',
      type=str,
      default="E:/ipeen/ipeen_data_total_num_cleaned.csv",
      help='Path to raw data'
    )
    parser.add_argument(
      '--user_id',
      type=int,
      default=-1,
      help='ID of user to recommend'
    )
    parser.add_argument(
      '--k',
      type=int,
      default=10,
      help='Number of most similar users to consider'
    )
    args = parser.parse_args()

    #Dataframe
    names = ['user_id', 'restaurant_id', 'rating', 'restaurant_reindex', 'user_reindex']
    df_reindex = pd.read_csv(args.filepath, sep=',', names=names, engine='python')
    #df_reindex.head()

    main(df_reindex, args.k, args.user_id)
