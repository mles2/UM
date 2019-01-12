from sklearn.svm import LinearSVC

class Svm:
    def __init__(self):
        print("SVM")
        self.clf = LinearSVC()#(gamma='scale', decision_function_shape='ovo')

    def learn(self, inputs, outputs):
        self.clf.fit(inputs, outputs)

    def predict_sample(self, sample):
        return self.clf.predict(sample.reshape(1,-1))
    
    def predict(self,sample):
        return self.clf.predict(sample)
