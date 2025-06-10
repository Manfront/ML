import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
data = iris.data
labels = iris.target
label_names = iris.target_names

pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

df = pd.DataFrame(data_reduced, columns=['PC1', 'PC2'])
df['Label'] = labels

plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i, label in enumerate(np.unique(labels)):
    plt.scatter(df[df['Label'] == label]['PC1'], df[df['Label'] == label]['PC2'],
                label=label_names[label], color=colors[i])
plt.title('PCA on Iris Dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid()
plt.show()
