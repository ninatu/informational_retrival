from pandas import DataFrame
import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor as DT


def relNormalize1(rel, max_rel=19):
    norm_rel = rel - min(rel)
    if max(norm_rel) != 0:
        norm_rel = norm_rel * max_rel / max(norm_rel) + 1
    return norm_rel

def relNormalize2(rel):
    uniq_rel = np.unique(rel)
    uniq_rel = sorted(uniq_rel)
    norm_rel = np.empty(rel.shape)
    for i, val in enumerate(uniq_rel):
        norm_rel[rel==val] = i + 1
    return norm_rel


class QueryData:
    def __init__(self, X, rel):
        self._X = np.copy(X)
        self._n = self._X.shape[0]
        self._rel = np.copy(rel)

        right_order = np.argsort(self._rel)[::-1]
        self._X = self._X[right_order]
        self._rel = self._rel[right_order]
        self._2_rel = 2 ** self._rel - 1
        self._log2_order = np.log2(np.array(range(self._n)) + 2)

    def maxDCG(self, T_NDCG):
        return sum(self._2_rel[:T_NDCG] / self._log2_order[:T_NDCG])

    def getX(self):
        return self._X

    def getY(self):
        return self._rel

    def getSwapNDCGMatrix(self, rel, T_NDCG):
        order = np.empty(rel.shape)
        sort_rel = sorted(zip(rel, range(rel.shape[0])), key=lambda x: x[0], reverse=True)
        order[list(map(lambda x: x[1], sort_rel))] = range(1, rel.shape[0] + 1)

        log2_order = np.log2(order + 1)
        elem_DCG = self._2_rel / log2_order
        matr_elem_DCG = np.tile(elem_DCG, (self._n, 1))
        matr_swap_elem_DCG = (self._2_rel.reshape((1, self._n)) / log2_order.reshape((self._n, 1)))

        lambda_mtr = - matr_elem_DCG - matr_elem_DCG.T + matr_swap_elem_DCG + matr_swap_elem_DCG.T
        no_null_swap = ((order <= T_NDCG).reshape((self._n, 1)) + (order <= T_NDCG).reshape((1, self._n))) > 0
        lambda_mtr = np.abs(lambda_mtr * no_null_swap)
        max_DCG = self.maxDCG(T_NDCG)
        if max_DCG != 0:
            return lambda_mtr / max_DCG
        else:
            return lambda_mtr

    def getNDCG(self, rel, T_NDCG):
        order = np.empty(rel.shape)
        sort_rel = sorted(zip(rel, range(rel.shape[0])), key=lambda x: x[0], reverse=True)
        order[list(map(lambda x: x[1], sort_rel))] = range(1, rel.shape[0] + 1)
        log2_order = np.log2(order + 1)
        elem_DCG = self._2_rel / log2_order
        DCG = sum(elem_DCG[order <= T_NDCG])
        max_DCG = self.maxDCG(T_NDCG)
        if max_DCG != 0:
            return DCG / max_DCG
        else:
            return DCG

    def getCountErrorPair(self, rel):
        _n = rel.shape[0]
        pairs = (rel.reshape((1, _n)) - rel.reshape((_n, 1))) > 0
        true_pairs = (self._rel.reshape((1, _n)) - self._rel.reshape((_n, 1))) > 0
        count = sum(sum((np.tril(pairs != true_pairs))))
        return count


class LambdaRankTrees:
    def __init__(self, learning_rate=0.5, n_estimators=1000, sigma=1, start_depth=5):
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        self._sigma = sigma
        self._start_depth = start_depth
        self._trees = None

    def _getTrainX(self, queries_data):
        n = 0
        m = queries_data[0].getX().shape[1]
        for query_data in queries_data:
            n += query_data._n

        X = np.empty((n, m), dtype=np.float64)
        Y = np.empty(n, dtype=np.float64)
        indexs = []
        cur_index = 0
        for query_data in queries_data:
            cur_n = query_data._n
            X[cur_index:cur_index + cur_n] = query_data.getX()
            Y[cur_index:cur_index + cur_n] = query_data.getY()
            indexs.append(range(cur_index, cur_index + cur_n))
            cur_index += cur_n
        return X, Y, indexs

    def _getGradient(self, queries_data, h, indexs_data, T_NDCG):
        g = np.empty(h.shape[0], dtype=np.float64)
        for i, indexs in enumerate(indexs_data):
            query_data = queries_data[i]
            rel = h[indexs]
            rel_n = rel.shape[0]
            lambda_matr = (- self._sigma / (1 + np.exp(self._sigma * (rel.reshape((1, rel_n)) -rel.reshape((rel_n, 1)))))) * query_data.getSwapNDCGMatrix(rel, T_NDCG)

            #filter_matr = (query_data._rel.reshape((1, rel_n)) - query_data._rel.reshape((rel_n, 1))) != 0
            #lambda_matr *= filter_matr
            tril = np.tril(lambda_matr)
            lambda_vector = np.sum(tril, axis=0) - np.sum(tril.T, axis=0)
            g[indexs] = lambda_vector
        return g

    def _getNDCG(self, queries_data, h, indexs_data, T_NDCG):
        ndcg = 0
        for i, indexs in enumerate(indexs_data):
            query_data = queries_data[i]
            rel = h[indexs]
            ndcg += query_data.getNDCG(rel, T_NDCG)
        return ndcg / len(indexs_data)

    def _countErrorPair(self, queries_data, h, indexs_data):
        count = 0
        for i, indexs in enumerate(indexs_data):
            query_data = queries_data[i]
            rel = h[indexs]
            count += query_data.getCountErrorPair(rel)
        return count

    def fit(self, queries_data, persent_valid=0.2, persent_train=0.1):
        random.seed(1234)
        random.shuffle(queries_data)
        count_valid = int(persent_valid * len(queries_data))
        data_valid = queries_data[:count_valid]
        data_train = queries_data[count_valid:]

        X_train, y_train, index_train = self._getTrainX(data_train)
        X_valid, y_valid, index_valid = self._getTrainX(data_valid)

        self._trees = []
        self._score_train = []
        self._score_valid = []

        h_train = np.zeros(X_train.shape[0])
        h_valid = np.zeros(X_valid.shape[0])
        for iteration in range(self._n_estimators):
            print(iteration, self._countErrorPair(data_train, h_train, index_train),
                  self._countErrorPair(data_valid, h_valid, index_valid),
                  self._getNDCG(data_train, h_train, index_train, 5),
                  self._getNDCG(data_valid, h_valid, index_valid, 5))

            g = self._getGradient(data_train, h_train, index_train, 5)
            d_tree = DT(max_depth=self._start_depth)
            d_tree.fit(X_train, -g)
            self._trees.append(d_tree)
            h_train += self._learning_rate * d_tree.predict(X_train)
            h_valid += self._learning_rate * d_tree.predict(X_valid)

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for tree in self._trees:

            y += self._learning_rate * tree.predict(X)
        return y

def loadData():
    trainPath = "../data/train.data.cvs"
    return DataFrame.from_csv(trainPath, index_col=False).as_matrix()

def loadSmallData():
    return np.load("small_train.npy")

def saveResults(queries, rels, namefile):
    uniq_queries = np.unique(queries)
    ans = np.empty((rels.shape[0], 2), dtype=np.int)
    for q in uniq_queries:
        rel = rels[queries == q]
        order = np.argsort(rel)[::-1] + 1
        ans[queries == q, 0] = order
        ans[queries == q, 1] = q
    df = DataFrame(ans, columns=["DocumentId","QueryId"])
    df.to_csv(open(namefile, "w"), index=False)

rowData = loadData()
queries = rowData[:, -1]
uniq_queries = np.unique(queries)
queries_train_data = []
for q in uniq_queries:
    xy = rowData[queries == q][:, :-1]
    queries_train_data.append(QueryData(xy[:, 1:], relNormalize2(xy[:, 0])))

lambdaRank = LambdaRankTrees(learning_rate=0.1, n_estimators=300, sigma=1, start_depth=5)
lambdaRank.fit(queries_train_data[:100],  persent_valid=0.2, persent_train=1)
