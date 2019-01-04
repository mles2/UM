from sklearn.neural_network import MLPClassifier

class NeuralNet:
    def __init__(self):
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(5, 2), random_state=1)

    def learn(self, inputs, outputs):
        self.clf.fit(inputs, outputs)

    def predict(self, sample):
        return self.predict(sample)