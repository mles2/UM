from sklearn.model_selection import train_test_split
from dtc import DecisionTree
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import numpy as np


class RemoveEigenfaceFutureSelector:

    def __init__(self, inputs_after_pca, target):
        self.inputs_after_pca = inputs_after_pca
        self.target = target
        self.number_of_vectors = inputs_after_pca.shape[1]

    def rank_features(self):
        ranking = []
        for vector_number in range(self.number_of_vectors):
            rank_of_vector = self.rank_one_vector(vector_number)
            ranking.append(rank_of_vector)
        return np.asarray(ranking)

    def rank_one_vector(self, vector_number):
        selector = [x for x in range(self.number_of_vectors) if x != vector_number]
        inputs_leave_one_out = self.inputs_after_pca[:, selector]

        accumulated_effectiveness = 0
        iterations = 5
        for i in range(iterations):
            x_train, x_test, y_train, y_test = train_test_split(inputs_leave_one_out, self.target, test_size=0.25)

            dtc = DecisionTree()
            dtc.learn(x_train, y_train)
            y_test_dtc_pred = dtc.predict(x_test)
            accumulated_effectiveness = accumulated_effectiveness + self.compute_effectiveness(y_test, y_test_dtc_pred)

        return accumulated_effectiveness/iterations

    def compute_effectiveness(self, ideal, real):
        accuracy = accuracy_score(ideal, real)
        loss = hamming_loss(ideal, real)
        # f1 = f1_score(ideal, real, average='macro') # warning
        # return (accuracy + (1-loss) + f1) / 3
        return (accuracy + (1-loss)) / 2
