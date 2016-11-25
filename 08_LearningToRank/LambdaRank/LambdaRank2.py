from pandas import DataFrame
import numpy as np
from math import log2
import random
from sklearn.tree import DecisionTreeRegressor as DT


def DCGScore(y_true, y_score, k=10, gains="exponential"):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def NDCGScore(y_true, y_score, k=10, gains="exponential"):
    best = DCGScore(y_true, y_true, k, gains)
    actual = DCGScore(y_true, y_score, k, gains)
    return actual / best

def deltaNDCGScore(y_true, y_score, k=10, gains="exponential"):
    max_DCG = DCGScore(y_true, y_true, k, gains)
    order = np.empty(y_score.shape)
    sort_y_score = sorted(zip(y_score, range(y_score.shape[0])), key=lambda x: x[0], reverse=True)
    order[list(map(lambda x: x[1], sort_y_score))] = range(1, y_score.shape[0] + 1)
    discounts = np.log2(order + 1)
    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")
    elem_DCG = gains / discounts
    _n = y_true.shape[0]
    matr_elem_DCG = np.tile(elem_DCG, (_n, 1))
    matr_swap_elem_DCG = gains.reshape((_n, 1)) / discounts.reshape((1, _n))

    lambda_mtr = - matr_elem_DCG - matr_elem_DCG.T + matr_swap_elem_DCG + matr_swap_elem_DCG.T
    no_null_swap = ((order <= k).reshape((_n, 1)) + (order <= k).reshape((1, _n))) > 0
    lambda_mtr = np.abs(lambda_mtr * no_null_swap)
    if max_DCG != 0:
        return lambda_mtr / max_DCG
    else:
        return lambda_mtr

def AverageNDCG(data, y_score, k):
    uniq_queries = np.unique(data[:, -1])
    ndcg = 0
    for q in uniq_queries:
        filt_q = (data[:, -1] == q)
        ndcg += NDCGScore(data[filt_q, 0], y_score[filt_q], k)
    return ndcg / uniq_queries.shape[0]

def CountErrorPair(data, y_score):
    uniq_queries = np.unique(data[:, -1])
    count_pair = 0
    for q in uniq_queries:
        filt_q = (data[:, -1] == q)
        y_true = data[filt_q, 0]
        y_pred = y_score[filt_q]

        _n = y_true.shape[0]
        true_pairs = (y_true.reshape((1, _n)) - y_true.reshape((_n, 1))) > 0
        pairs = (y_pred.reshape((1, _n)) - y_pred.reshape((_n, 1))) > 0
        count = sum(sum((np.tril(pairs != true_pairs))))
        count_pair += count

    return count_pair

def relNormalize2(rel):
    uniq_rel = np.unique(rel)
    uniq_rel = sorted(uniq_rel)
    norm_rel = np.empty(rel.shape)
    for i, val in enumerate(uniq_rel):
        norm_rel[rel==val] = i + 1
    return norm_rel * 5


class LambdaRankForest:
    def __init__(self, learning_rate, n_estimators, sigma, start_depth):
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        self._sigma = sigma
        self._start_depth = start_depth
        self._trees = []

    def fit(self, DATA, persent_valid, normalize, T_NDCG):
        all_queries = DATA[:, -1]
        uniq_queries = np.unique(all_queries)
        for q in uniq_queries:
            filt = (all_queries == q)
            DATA[filt, 0] = normalize(DATA[filt, 0])
            order = np.argsort(DATA[filt, 0])[::-1]
            DATA[filt] = DATA[filt][order]

        random.shuffle(uniq_queries)
        count_valid = int(persent_valid * uniq_queries.shape[0])

        valid_queries = uniq_queries[:100]
        train_queries = uniq_queries[100:200]
        DATA_valid = np.concatenate([DATA[all_queries == q] for q in valid_queries], axis=0)
        DATA_train = np.concatenate([DATA[all_queries == q] for q in train_queries], axis=0)
        v_queries = DATA_valid[:, -1]
        t_queries = DATA_train[:, -1]

        self._trees = []
        h_train = np.zeros(DATA_train.shape[0])
        h_valid = np.zeros(DATA_valid.shape[0])
        iteration = 0
        while True:
            print(iteration,
                 CountErrorPair(DATA_train, h_train),
                 CountErrorPair(DATA_valid, h_valid),

                 AverageNDCG(DATA_train, h_train, 20),
                 AverageNDCG(DATA_valid, h_valid, 20),
                 AverageNDCG(DATA_train, h_train, 5),
                 AverageNDCG(DATA_valid, h_valid, 5))

            grad = np.zeros(h_train.shape)
            for q in uniq_queries:
                filter_query = (t_queries == q)
                h = h_train[filter_query]
                y = DATA_train[filter_query, 0]
                p = 1 + np.exp((h.reshape((1, h.shape[0])) -h.reshape((h.shape[0], 1))) * self._sigma)
                g = (- self._sigma / p) #* deltaNDCGScore(y, h, T_NDCG)
                filt = ((y.reshape((y.shape[0], 1)) - y.reshape((1, y.shape[0]))) != 0)
                g *= filt

                g = np.tril(g)
                xx = np.sum(g, axis=0) - np.sum(g.T, axis=0)
                grad[filter_query] = np.sum(g, axis=0) - np.sum(g.T, axis=0)

            new_tree = DT(max_depth=self._start_depth)
            new_tree.fit(DATA_train[:, 1:-1], -grad)
            self._trees.append(new_tree)
            h_train += self._learning_rate * new_tree.predict(DATA_train[:, 1:-1])
            h_valid += self._learning_rate * new_tree.predict(DATA_valid[:, 1:-1])

            iteration += 1
            if (iteration == self._n_estimators):
                break

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self._trees:
            y_pred += self._learning_rate * tree.predict(X)
        return y_pred

def loadData():
    trainPath = "../data/train.data.cvs"
    return DataFrame.from_csv(trainPath, index_col=False).as_matrix()

rowData = loadData()
estimator = LambdaRankForest(learning_rate=0.1, n_estimators=100, sigma=1, start_depth=3)
estimator.fit(DATA=rowData, persent_valid=0.5, normalize=relNormalize2, T_NDCG=30)
