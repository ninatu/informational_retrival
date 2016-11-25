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
    matr_elem_DCG = np.tile(elem_DCG, (_n, 1)).T
    matr_swap_elem_DCG = gains.reshape((_n, 1)) / discounts.reshape((1, _n))

    lambda_mtr = - matr_elem_DCG - matr_elem_DCG.T + matr_swap_elem_DCG + matr_swap_elem_DCG.T
    no_null_swap = ((order <= k).reshape((_n, 1)) + (order <= k).reshape((1, _n))) > 0
    lambda_mtr = np.abs(lambda_mtr * no_null_swap)
    if max_DCG != 0:
        return lambda_mtr / max_DCG
    else:
        return lambda_mtr


def AverageNDCG(data, y_score, k):
    ndcg = 0
    for (indexs, y_true) in data:
        ndcg += NDCGScore(y_true, y_score[indexs], k)
    return ndcg / len(data)


def CountErrorPair(data, y_score):
    count_pair = 0
    for (indexs, y_true) in data:
        y_pred = y_score[indexs]
        _n = y_pred.shape[0]

        pairs = (y_pred.reshape((_n, 1)) - y_pred.reshape((1, _n))) > 0
        true_pairs = (y_true.reshape((_n, 1)) - y_true.reshape((1, _n))) > 0
        count = sum(sum(pairs != true_pairs))
        count_pair += count
    return count_pair

def relNormalize2(rel):
    uniq_rel = np.unique(rel)
    uniq_rel = sorted(uniq_rel)
    norm_rel = np.empty(rel.shape)
    for i, val in enumerate(uniq_rel):
        norm_rel[rel==val] = i + 1
    return norm_rel * 5


class LambdaRank:
    def __init__(self, learning_rate, n_estimators, sigma, start_depth):
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        self._sigma = sigma
        self._start_depth = start_depth
        self._trees = []

    def _createSet(self, DATA, queries, normalize):
        all_queries = DATA[:, -1]
        seq_x = []
        data = []
        last_index = 0
        for q in queries:
            X = DATA[all_queries == q, 1:-1]
            seq_x.append(X)
            data.append((range(last_index, last_index + X.shape[0]), normalize(DATA[all_queries == q, 0])))
            last_index += X.shape[0]
        return data, np.concatenate(seq_x, axis=0)

    def fit(self, DATA, persent_valid, normalize, T_NDCG):
        all_queries = DATA[:, -1]
        uniq_queries = np.unique(all_queries)
        #random.shuffle(uniq_queries)
        uniq_queries = uniq_queries[:100]

        """
        for q in uniq_queries:
            filt = (all_queries == q)
            DATA[filt, 0] = normalize(DATA[filt, 0])
            order = np.argsort(DATA[filt, 0])[::-1]
            DATA[filt] = DATA[filt][order]
        """

        count_valid = int(persent_valid * uniq_queries.shape[0])
        valid_queries = uniq_queries[:count_valid]
        train_queries = uniq_queries[count_valid:]
        """
        DATA_valid = np.concatenate([DATA[all_queries == q] for q in valid_queries], axis=0)
        DATA_train = np.concatenate([DATA[all_queries == q] for q in train_queries], axis=0)
        t_queries = DATA_train[:, -1]
        y_true = DATA_train[:, 0]
        X_train = DATA_train[:, 1:-1]
        """
        data_train, X_train = self._createSet(DATA, train_queries, normalize)
        data_valid, X_valid = self._createSet(DATA, valid_queries, normalize)

        self._trees = []
        h_train = np.zeros(X_train.shape[0])
        h_valid = np.zeros(X_valid.shape[0])

        iteration = 0
        while True:
            grad = np.zeros(h_train.shape)
            for (indexs, y) in data_train:
                h = h_train[indexs]
                _n = h.shape[0]

                delta_h = h.reshape((_n, 1)) - h.reshape((1, _n))
                sign_h = np.sign(y.reshape((_n, 1)) - y.reshape((1, _n)))
                p = 1 + np.exp(self._sigma * delta_h * sign_h)
                # * deltaNDCGScore(y, h, T_NDCG)
                xx = np.sum(self._sigma * sign_h / p, axis=1)
                grad[indexs] = xx

            new_tree = DT(max_depth=self._start_depth)
            new_tree.fit(X_train, grad)
            self._trees.append(new_tree)

            h_train += self._learning_rate * new_tree.predict(X_train)
            h_valid += self._learning_rate * new_tree.predict(X_valid)

            print(iteration,
                  CountErrorPair(data_train, h_train),
                  CountErrorPair(data_valid, h_valid),
                  AverageNDCG(data_train, h_train, 5),
                  AverageNDCG(data_valid, h_valid, 5),
                  np.linalg.norm(-grad))
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

estimator = LambdaRank(learning_rate=0.1, n_estimators=100, sigma=1, start_depth=3)
estimator.fit(DATA=rowData, persent_valid=0.2, normalize=relNormalize2, T_NDCG=30)
