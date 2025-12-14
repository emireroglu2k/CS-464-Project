import numpy as np
from collections import Counter

class CustomKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []
        
        for i, x in enumerate(X_test):
            # Euclidean Distance between x and all examples in X_train
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # sort the distances and get the first k neighbors
            k_indices = np.argsort(distances)[:self.k]

            # labels of the k nearest neighbors
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            # find the most common label
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
            
            # print progress every 100 images
            if i % 100 == 0:
                print(f"Processed {i}/{len(X_test)} samples...")

        return np.array(predictions)