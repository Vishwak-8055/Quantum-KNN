from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run_classical_knn(X_train, X_test, y_train, y_test):

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return accuracy