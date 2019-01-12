from sklearn.neural_network import MLPClassifier

class NeuralNet:
    def __init__(self, epochs):
        print("MLP")
        self.clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(25, 10), random_state=1, max_iter = epochs)#lbfgs solver

    def learn(self, inputs, outputs):
        self.clf.fit(inputs, outputs)

    def predict_sample(self, sample):
        return self.clf.predict(sample.reshape(1,-1))
    
    def predict(self,sample):
        return self.clf.predict(sample)