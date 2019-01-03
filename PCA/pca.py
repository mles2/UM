from sklearn import datasets
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from math import sqrt
import cv2


faces= datasets.fetch_olivetti_faces()
print(faces.data.shape)

def display_image(image):
    plt.figure()
    plt.imshow(image.reshape(faces.images[0].shape), cmap=plt.cm.bone)

def generate_eigenfaces(number_of_eigenfaces):
    pca = decomposition.PCA(n_components=number_of_eigenfaces, whiten=True)
    return pca.fit(faces.data)

pca = generate_eigenfaces(5)


example = pca.transform(faces.data[1].reshape(1,-1))
print(example.shape)
inverse = pca.inverse_transform(example)
print(inverse.shape)

display_image(faces.data[1])
display_image(inverse)

#plt.show()