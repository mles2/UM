import numpy as np


def get_values_at_indexes(inputs, indexes):
    number_of_vectors = indexes.shape[0]
    chosen_values = np.zeros((inputs.shape[0], number_of_vectors))

    for n, index in zip(range(number_of_vectors), indexes):
        chosen_values[:, n] = inputs[:, index]

    return chosen_values


def get_lowest_values_indexes(values_vector, number_to_get):
    return values_vector.argsort()[:number_to_get]


class FutureSelectionService:

    @staticmethod
    def get_best_features(inputs, feature_selector_ranking, number_to_get):
        indexes = get_lowest_values_indexes(feature_selector_ranking, number_to_get)
        return get_values_at_indexes(inputs, indexes)
