from Xpreprocessing import PreProcessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np



def lg(X_train, X_test, Y_train, Y_test):

    classifier = LogisticRegression(penalty="l1",random_state=2,C=5,solver="saga")
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)

    #print(np.concatenate((y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

    cm = confusion_matrix(Y_test, y_pred)
    #print(cm)
    print("LogisticRegression Accuracy : ", accuracy_score(Y_test, y_pred) *100,"%")
    return


def knn(X_train, X_test, Y_train, Y_test):
    classifier = KNeighborsClassifier(n_neighbors=13, metric='minkowski', p=2)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(Y_test, y_pred)
    # print(cm)
    print("KNeighborsClassifier Accuracy : ", accuracy_score(Y_test, y_pred) * 100, "%")

    return

def dt(X_train, X_test, Y_train, Y_test):
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 5)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(Y_test, y_pred)
    # print(cm)
    print("DecisionTreeClassifier Accuracy : ", accuracy_score(Y_test, y_pred) * 100, "%")

    return

def main():
    X_train, X_test, Y_train, Y_test = PreProcessing()

    lg( X_train, X_test, Y_train, Y_test)
    knn( X_train, X_test, Y_train, Y_test)
    dt( X_train, X_test, Y_train, Y_test)



if __name__ == "__main__":
     main()