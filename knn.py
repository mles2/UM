from sklearn.neighbors import KNeighborsClassifier

class Knn:
    def __init__(self, number_of_neighbors):
        #print("KNN")
        self.clf = KNeighborsClassifier(number_of_neighbors)

    def learn(self, inputs, outputs):
        self.clf.fit(inputs, outputs)

    def predict_sample(self, sample):
        return self.clf.predict(sample.reshape(1,-1))
    
    def predict(self,sample):
        return self.clf.predict(sample)
