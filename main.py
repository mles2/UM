from mlp import NeuralNet
from pca import PCAManipulator
from knn import Knn
from svm import Svm
from sklearn import datasets
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, classification_report
from matplotlib import pyplot as plt
from future_selection_service import FutureSelectionService as fss
from remove_eigenface_feature_selector import RemoveEigenfaceFutureSelector
from evalueation_scores import EvaluationScores


def display_image(image, shape):
    plt.figure()
    plt.imshow(image.reshape(shape), cmap=plt.cm.bone)


def compute_metrics(accuracy_type, ideal, real):
    accs = accuracy_score(ideal, real)
    hl = hamming_loss(ideal, real)
    f1s = f1_score(ideal, real, average='macro')

    # print("    ", accuracy_type, ":")
    # print("         accuracy score: ", accs)
    # print("         loss: ", hl)
    # print("         F1 score: ", f1s)
    # print("         classification report: ")
    # print(classification_report(ideal, real))
    return accs, hl, f1s


def evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    classifier.learn(X_train, y_train)
    y_train_mlp_pred = classifier.predict(X_train)
    y_test_mlp_pred = classifier.predict(X_test)
    accs_train, hl_train, f1s_train = compute_metrics("Train", y_train, y_train_mlp_pred)
    accs_test, hl_test, f1s_test = compute_metrics("Test", y_test, y_test_mlp_pred)
    return EvaluationScores(accs_train, hl_train, f1s_train, accs_test, hl_test, f1s_test)

def evaluateNeuralNet(X_train, X_test, y_train, y_test):
    mlp = NeuralNet(10000)
    return evaluate_classifier(mlp, X_train, X_test, y_train, y_test)

def evaluateKnn(X_train, X_test, y_train, y_test):
    knn = Knn(2)
    return evaluate_classifier(knn, X_train, X_test, y_train, y_test)

def evaluateSvm(X_train, X_test, y_train, y_test):
    svm = Svm()
    return evaluate_classifier(svm, X_train, X_test, y_train, y_test)

def perform_experiment_with_selected_features(inputs_from_feature_selection, dataset):
    feature_selection_NN = EvaluationScores()
    feature_selection_Knn = EvaluationScores()
    feature_selection_Svm = EvaluationScores()
    for train_index, test_index in cv5x2.split(inputs_from_feature_selection, dataset.target):
        X_train, X_test = inputs_from_feature_selection[train_index], inputs_from_feature_selection[test_index]
        y_train, y_test = dataset.target[train_index], dataset.target[test_index]
        feature_selection_NN = feature_selection_NN + evaluateNeuralNet(X_train, X_test, y_train, y_test)
        feature_selection_Knn = feature_selection_Knn + evaluateKnn(X_train, X_test, y_train, y_test)
        feature_selection_Svm = feature_selection_Svm + evaluateSvm(X_train, X_test, y_train, y_test)
    feature_selection_NN = feature_selection_NN / (N_REPEATS * N_FOLDS)
    feature_selection_Knn = feature_selection_Knn / (N_REPEATS * N_FOLDS)
    feature_selection_Svm = feature_selection_Svm / (N_REPEATS * N_FOLDS)
    return feature_selection_NN, feature_selection_Knn, feature_selection_Svm

NUMBER_OF_FEATURES = 50
N_FOLDS = 2
N_REPEATS = 5
cv5x2 = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=36851234)

dataset = datasets.fetch_olivetti_faces()
pca = PCAManipulator(100, dataset)

# X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25)
# print(X_train.shape)

inputs_after_pca = pca.transform(dataset.data)
print("inputs after pca shape")
print(inputs_after_pca.shape)

# TODO
# wybranie jednego i policzenie drzewem, /DONE
# kroswalidacja, /DONE
# wybieranie danych na podstawie istotności /DONE
# zebranie danych z eksperymentu, /DONE
# redakcja artykułu
# testy parowe F test

outputs = dataset.target
print(outputs.shape)

# MutualInformation Selection
mutual_information_feature_selection = mutual_info_classif(inputs_after_pca, outputs, discrete_features='auto',
                                                           n_neighbors=3, copy=True, random_state=None)
inputs_mutual_information = fss.get_best_features(inputs_after_pca, mutual_information_feature_selection,
                                                  NUMBER_OF_FEATURES)
mutual_information_NN, mutual_information_Knn, mutual_information_Svm = perform_experiment_with_selected_features(inputs_mutual_information, dataset)


# RFE Information Selection
rfe_feature_selection = RFE(inputs_after_pca, outputs)
estimator = SVR(kernel="linear")
rfe_feature_selection = RFE(estimator, NUMBER_OF_FEATURES, step=1)
rfe_feature_selection = rfe_feature_selection.fit(inputs_after_pca, outputs).ranking_
inputs_rfe = inputs_after_pca[:, rfe_feature_selection == 1]
rfe_NN, rfe_Knn, rfe_Svm = perform_experiment_with_selected_features(inputs_rfe, dataset)

# Remove eigenface selection
refs = RemoveEigenfaceFutureSelector(inputs_after_pca, dataset.target)
remove_eigenface_feature_selection = refs.rank_features()
inputs_remove_eigenface = fss.get_best_features(inputs_after_pca, remove_eigenface_feature_selection,
                                                NUMBER_OF_FEATURES)
remove_eigenface_NN, remove_eigenface_Knn, remove_eigenface_Svm = perform_experiment_with_selected_features(inputs_remove_eigenface, dataset)
