from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def __init__(self):
        self.clf = DecisionTreeClassifier()

    def learn(self, inputs, outputs):
        self.clf.fit(inputs, outputs)

    def predict_sample(self, sample):
        return self.clf.predict(sample.reshape(1, -1))

    def predict(self, sample):
        return self.clf.predict(sample)
