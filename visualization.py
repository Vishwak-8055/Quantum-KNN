import matplotlib.pyplot as plt

def plot_accuracy(classical_acc, quantum_acc):

    models = ["Classical KNN", "Quantum KNN"]
    accuracy = [classical_acc, quantum_acc]

    plt.bar(models, accuracy)

    plt.ylabel("Accuracy")
    plt.title("Classical vs Quantum KNN")

    return plt