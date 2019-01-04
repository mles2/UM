from sklearn import decomposition
import cv2

class PCAManipulator:

    def __init__(self, number_of_eigenfaces, dataset):
        self.number_of_eigenfaces = number_of_eigenfaces
        self.dataset = dataset
        self.pca = self.generate_eigenfaces(self.number_of_eigenfaces)

    def get_image_shape(self):
        return self.dataset.images[0].shape

    def generate_eigenfaces(self, number_of_eigenfaces):
        pca = decomposition.PCA(n_components=number_of_eigenfaces, whiten=False)
        return pca.fit(self.dataset.data)

    def transform_sample(self, sample):
        return self.pca.transform(sample.reshape(1,-1))

    def transform(self, sample):
        return self.pca.transform(sample)

    def reconstruct_sample(self, sample):
        return self.pca.inverse_transform(sample)
