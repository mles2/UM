from mlp import NeuralNet
from pca import PCAManipulator
from knn import Knn
from svm import Svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

def display_image(image, shape):
        plt.figure()
        plt.imshow(image.reshape(shape), cmap=plt.cm.bone)

dataset = datasets.fetch_olivetti_faces()
pca = PCAManipulator(100, dataset)

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=42)

print(X_train.shape)

inputs_after_pca = pca.transform(dataset.data)
print(inputs_after_pca.shape)

outputs = dataset.target
print(outputs.shape)

mlp = NeuralNet(1000)
mlp.learn(X_train, y_train)
y_train_mlp_pred = mlp.predict(X_train)
y_test_mlp_pred = mlp.predict(X_test)
print("MLP Learn accuracy score:", accuracy_score(y_train, y_train_mlp_pred))
print("MLP Test accuracy score:", accuracy_score(y_test, y_test_mlp_pred))

knn = Knn(2)
knn.learn(X_train, y_train)
y_train_knn_pred = knn.predict(X_train)
y_test_knn_pred = knn.predict(X_test)
print("KNN Learn accuracy score:", accuracy_score(y_train, y_train_knn_pred))
print("KNN Test accuracy score:", accuracy_score(y_test, y_test_knn_pred))

svm = Svm()
svm.learn(X_train, y_train)
y_train_svm_pred = svm.predict(X_train)
y_test_svm_pred = svm.predict(X_test)
print("SVM Learn accuracy score:", accuracy_score(y_train, y_train_svm_pred))
print("SVM Test accuracy score:", accuracy_score(y_test, y_test_svm_pred))

# transformed = pca.transform_sample(dataset.data[10])
# reconstructed = pca.reconstruct_sample(transformed)
# display_image(dataset.data[10], pca.get_image_shape())
# display_image(reconstructed, pca.get_image_shape())
plt.show()