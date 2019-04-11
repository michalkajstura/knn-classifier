import numpy as np


class KNearestNeighbors:
    def __init__(self, num_of_classes,  n_neighbors=5, metric='hamming'):
        self.num_of_classes = num_of_classes
        self.n_neigbors = n_neighbors

        if metric not in ['hamming']:
            raise Exception(metric + ' metric is not known')
        self.metric = metric

    def predict(self, x, x_train, y_train):
        distances = None
        if self.metric == 'hamming':
            distances = self.hamming_distance(x, x_train)

        labels = self.sort_train_labels_knn(distances, y_train)
        p_y_x = self.p_y_x_knn(labels, self.n_neigbors)
        y_pred = np.argmax(p_y_x, axis=1)
        return y_pred

    @staticmethod
    def hamming_distance(x, x_train):
        """
        Hamming distance between documents

        :param x: array of new documents N1xD
        :param x_train: array of already classified documents N2xD
        :return: matrix of hamming  distances N1xN2
        """

        hamming_similarity = x.dot(x_train.T) + (1 - x).dot(1 - x_train.T)
        dimentions = x.shape[1]
        return dimentions - hamming_similarity

    @staticmethod
    def sort_train_labels_knn(dist, y):
        """
        :param dist matrix of hamming  distances N1xN2
        :param y: vector of known labels N2
        :return: matrix of labels sorted in ascending order
        """
        idx_sorted = np.argsort(dist, axis=1, kind='mergesort')
        return y[idx_sorted]

    def p_y_x_knn(self, y, k):
        """
        Computes probablility distribution p(y|x) for each class from test set

        :param y: matrix of labels sorted in ascending order
        :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
        """

        nearest_neighbors = y[:, :k]

        # Sum of count of all classes throught 0 to M, divided by total number of classes (M)
        p_y_x_t = np.array(
            [(nearest_neighbors == m).sum(axis=1) / (self.num_of_classes + 1)
             for m in range(0, self.num_of_classes)])
        return p_y_x_t.T  # result is trasposed