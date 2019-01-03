from sklearn import datasets
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from math import sqrt
import cv2

def display_image(image, shape):
        plt.figure()
        plt.imshow(image.reshape(shape), cmap=plt.cm.bone)

class PCAManipulator:

    def __init__(self, number_of_eigenfaces, dataset):
        self.number_of_eigenfaces = number_of_eigenfaces
        self.dataset = dataset
        self.pca = self.generate_eigenfaces(self.number_of_eigenfaces)

    def get_image_shape(self):
        return self.dataset.images[0].shape

    def generate_eigenfaces(self, number_of_eigenfaces):
        pca = decomposition.PCA(n_components=number_of_eigenfaces, whiten=True)
        return pca.fit(self.dataset.data)

    def transform_sample(self, sample):
        return self.pca.transform(sample.reshape(1,-1))

    def reconstruct_sample(self, sample):
        return self.pca.inverse_transform(sample)


dataset = datasets.fetch_olivetti_faces()
pca = PCAManipulator(100, dataset)
transformed = pca.transform_sample(dataset.data[10])
reconstructed = pca.reconstruct_sample(transformed)

display_image(dataset.data[10], pca.get_image_shape())
display_image(reconstructed, pca.get_image_shape())

plt.show()