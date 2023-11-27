import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def logreg(file):
    # Reading the data
    data = pd.read_csv(file)

    # Extract feature(s) (x) and target variable (y)
    x = data[['Pclass', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']]
    y = data['Survived']

    # Feature scaling
    def feature_scaling(a):
        a = (a - a.mean()) / a.std()
        return a

    x = feature_scaling(x)

    # Convert to NumPy arrays
    x_array = x.values
    y_array = y.values

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def logistic_regression(x, y, num_iterations, learning_rate):
        num_samples, num_features = x.shape
        intercept = np.ones((num_samples, 1))
        x = np.concatenate((intercept, x), axis=1)
        weights = np.zeros(num_features + 1)
        
        for i in range(num_iterations):
            scores = np.dot(x, weights)
            predictions = sigmoid(scores)
            gradient = np.dot(x.T, (predictions - y)) / num_samples
            weights -= learning_rate * gradient
        
        return weights

    # Train logistic regression model using your function
    num_iterations = 1000
    learning_rate = 0.007
    weights = logistic_regression(x_array, y_array, num_iterations, learning_rate)

    # Predict probabilities on the entire dataset
    intercept_all = np.ones((x_array.shape[0], 1))
    x_all = np.concatenate((intercept_all, x_array), axis=1)
    predicted_probs = sigmoid(np.dot(x_all, weights))

    def predict_labels(x, weights, threshold=0.5):
        # Add bias term to features
        intercept = np.ones((x.shape[0], 1))
        x = np.concatenate((intercept, x), axis=1)
        
        # Predict probabilities
        predicted_probs = sigmoid(np.dot(x, weights))
        
        # Convert probabilities to binary predictions using a threshold
        predicted_labels = (predicted_probs >= threshold).astype(int)
        
        return predicted_labels

    def accuracy(y_true, y_pred):
        # Calculate accuracy
        correct_predictions = np.sum(y_true == y_pred)
        total_samples = len(y_true)
        accuracy = correct_predictions / total_samples
        
        return accuracy

    # Use the trained weights to make predictions on the training data
    predicted_labels_train = predict_labels(x_array, weights)

    # Calculate accuracy on the training data
    accuracy_train = accuracy(y_array, predicted_labels_train)
    print(f" Accuracy: {accuracy_train * 100:.2f}%")
logreg("/Users/abhudaysingh/Downloads/test (1).csv")
