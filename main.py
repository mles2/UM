from mlp import NeuralNet
from pca import PCAManipulator
from knn import Knn
from svm import Svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, classification_report
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def display_image(image, shape):
    plt.figure()
    plt.imshow(image.reshape(shape), cmap=plt.cm.bone)

def compute_metrics(accuracy_type, ideal, real):
    print("    ", accuracy_type,":")
    print("         accuracy score: ", accuracy_score(ideal, real))
    print("         loss: ", hamming_loss(ideal, real))
    print("         F1 score: ", f1_score(ideal, real, average='macro'))
    print("         classification report: ")
    print(classification_report(ideal, real))

dataset = datasets.fetch_olivetti_faces()
pca = PCAManipulator(100, dataset)

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25)

print(X_train.shape)

inputs_after_pca = pca.transform(dataset.data)
print(inputs_after_pca.shape)

#TODO
#wybranie jednego i policzenie drzewem,
#kroswalidacja,
#wybieranie danych na podstawie istotności
#zebranie danych z eksperymentu,
#redakcja artykułu

outputs = dataset.target
print(outputs.shape)

#MutualInformation Selection
mutual_information_feature_selection = mutual_info_classif(inputs_after_pca, outputs, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
print(mutual_information_feature_selection.shape, mutual_information_feature_selection)

#RFE Information Selection
rfe_feature_selection = RFE(inputs_after_pca, outputs)
estimator = SVR(kernel="linear")
rfe_feature_selection = RFE(estimator, 5, step=1)
rfe_feature_selection = rfe_feature_selection.fit(inputs_after_pca, outputs).ranking_
print(rfe_feature_selection.shape, rfe_feature_selection)

#Decision tree clasifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_train_dtc_pred = dtc.predict(X_train)
y_test_dtc_pred = dtc.predict(X_test)
print("Decision tree")
compute_metrics("Train", y_train, y_train_dtc_pred)
compute_metrics("Test", y_test, y_test_dtc_pred)


mlp = NeuralNet(10000)
mlp.learn(X_train, y_train)
y_train_mlp_pred = mlp.predict(X_train)
y_test_mlp_pred = mlp.predict(X_test)
print("MLP")
compute_metrics("Train", y_train, y_train_mlp_pred)
compute_metrics("Test", y_test, y_test_mlp_pred)

knn = Knn(2)
knn.learn(X_train, y_train)
y_train_knn_pred = knn.predict(X_train)
y_test_knn_pred = knn.predict(X_test)
print("KNN")
compute_metrics("Train", y_train, y_train_knn_pred)
compute_metrics("Test", y_test, y_test_knn_pred)

svm = Svm()
svm.learn(X_train, y_train)
y_train_svm_pred = svm.predict(X_train)
y_test_svm_pred = svm.predict(X_test)
print("SVM")
compute_metrics("Train", y_train, y_train_svm_pred)
compute_metrics("Test", y_test, y_test_svm_pred)

# transformed = pca.transform_sample(dataset.data[10])
# reconstructed = pca.reconstruct_sample(transformed)
# display_image(dataset.data[10], pca.get_image_shape())
# display_image(reconstructed, pca.get_image_shape())
plt.show()