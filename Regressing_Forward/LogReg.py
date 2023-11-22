import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the data
data = pd.read_csv("/Users/abhudaysingh/Downloads/weatherAUS.csv")

# Extract feature(s) (x) and target variable (y)
#x = data[['Rainfall', 'MinTemp', 'MaxTemp', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']]
x = data[['Rainfall', 'MinTemp', 'MaxTemp', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']]

y = data['RainTomorrow']

# Convert 'Yes' and 'No' to 1 and 0
y = y.map({'Yes': 1, 'No': 0})

# Feature scaling
x = (x - x.mean()) / x.std()

# Add a column of ones to the features for the intercept term
x['intercept'] = 1

# Convert to NumPy arrays
x_array = x.values
y_array = y.values

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, theta, learning_rate, num_iterations, lambda_):
    m, n = X.shape
    
    for _ in range(num_iterations):
        z = np.dot(X, theta)
        f_mc = sigmoid(z)
        err = f_mc - y

        dm = np.dot(X.T, err) / m
        dc = np.sum(err) / m

        # Add regularization term to the gradient with respect to the slope coefficients
        dm[1:] += (lambda_ / m) * theta[1:]

        # Update parameters using the learning rate
        theta -= learning_rate * dm
        theta[0] -= learning_rate * dc

        # Check for nan in weights
        if np.isnan(np.sum(theta)):
            print("Weights became nan. Stopping optimization.")
            break

    return theta

# Initial guesses and hyperparameters
initial_theta = np.zeros(x_array.shape[1])
learning_rate = 0.0001
num_iterations = 15000
lambda_ = 0.1

# Perform gradient descent
theta = gradient_descent(x_array, y_array, initial_theta, learning_rate, num_iterations, lambda_)
#print(x_array) Working Proper,y
print(theta) #Problem Here
# Predict probabilities
predicted_probs = sigmoid(np.clip(np.dot(x_array, theta), -500, 500))  # Clip values to prevent overflow
#print(predicted_probs) problem here due to theta
# Plot the predicted probabilities against the actual values
plt.scatter(predicted_probs, y, color="blue")
plt.xlabel("Predicted Probabilities")
plt.ylabel("Actual Values")
plt.title("Logistic Regression Predictions")
plt.show()
