import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data = np.random.rand(100)
labels = ["Class1" if x <= 0.5 else "Class2" for x in data[:50]]

def euclidean_distance(x1, x2):
    return abs(x1 - x2)

def knn_classifier(train_data, train_labels, test_point, k):
    distances = [(euclidean_distance(test_point, train_data[i]), train_labels[i]) for i in range(len(train_data))]
    distances.sort(key=lambda x: x[0])
    k_labels = [label for _, label in distances[:k]]
    return Counter(k_labels).most_common(1)[0][0]

train_data = data[:50]
test_data = data[50:]
k_values = [1, 2, 3, 4, 5, 20, 30]
results = {}

for k in k_values:
    classified = [knn_classifier(train_data, labels, point, k) for point in test_data]
    results[k] = classified
    for i, label in enumerate(classified, start=51):
        print(f"x{i} (value: {test_data[i-51]:.4f}) -> {label}")
    print()

for k in k_values:
    classified = results[k]
    class1 = [test_data[i] for i in range(len(test_data)) if classified[i] == "Class1"]
    class2 = [test_data[i] for i in range(len(test_data)) if classified[i] == "Class2"]
    plt.figure(figsize=(10, 6))
    plt.scatter(train_data, [0] * len(train_data), c=["blue" if l == "Class1" else "red" for l in labels], label="Train", marker="o")
    plt.scatter(class1, [1] * len(class1), c="blue", label="Class1 (Test)", marker="x")
    plt.scatter(class2, [1] * len(class2), c="red", label="Class2 (Test)", marker="x")
    plt.title(f"k-NN Results for k = {k}")
    plt.xlabel("Data")
    plt.ylabel("Level")
    plt.legend()
    plt.grid(True)
    plt.show()
