#import for data manipulation
import numpy as np
import pandas as pd
#for PCA and kmeans clustering
from sklearn.decomposition import PCA
from sklearn import preprocessing
#we use the label encoder to transform strings into numeric values
from sklearn.preprocessing import LabelEncoder
#for data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/abhudaysingh/Downloads/Clustering_dataset.zip")
data
#we drop the a few columns from the dataset and new dataset is X
X = data.drop('Date', axis=1)
X=X.drop('Location',axis=1)
label_encoder = LabelEncoder()
#we use label encoder to emcode the categorical column which have strings 
X['WindGustDir'] = label_encoder.fit_transform(X['WindGustDir'])
X['WindDir9am'] = label_encoder.fit_transform(X['WindDir9am'])
X['WindDir3pm'] = label_encoder.fit_transform(X['WindDir3pm'])

#replace all NaN values with 0
X=X.fillna(0)
X=X.replace('Yes','1')
X=X.replace('No','0')
#X
#we create a corellation matrix 
correlation_matrix = X.corr()
# we display it as a heatmap using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()
#note: correlation matrix is helpful in multivariable analysis
#it is helpful in understanding relations between variables

#we print the first few rows of the dataset
#to anaylse/inspect the structure and content of the dataframe 
print(X.head())

#shows the number of rows and columns of the dataset
print(X.shape)

#PCA
#we do this pca here to get all the principal components in our dataset
#we standardize the data by subtracting the mean and dividing by the std. deviation for each feature
#we use X.T to transpose the dataset to ensure that pca happens along columns and not features
X_scaled = preprocessing.scale(X.T)
#we set num of principal components to the no. of rows and columns in the dataframe
#we do this to limit the no. of principal components
num_components = min(X.shape[0], X.shape[1])  
#we initialize pca object from scikit learn with the specific no. of components
pca = PCA(num_components) #n_components=num components
#we perform pca on the standardized data
X_pca = pca.fit_transform(X_scaled)
# The fit_transform method fits the PCA model and transforms the input data into the principal components
#X_pca contains the dataset in the reduced dimensional space defined by the principal components

#we now plot to visualize the variance of each principal component

#we calculate the percentage variance of each of the principal comps.
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)    
#we create labels for the principal comps. PC1,PC2.....
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

#we plot the cumulative explained variance against no. of components
plt.plot(range(1,22),pca.explained_variance_ratio_.cumsum(),marker = 'o')
plt.xlabel("no. of components")
plt.ylabel("cumulative explained variance")
plt.show()

#we again perform pca with num of comps set to 2 (we got this from the scree plot)
#we want to reduce the dimensionality to 2 prinicipal components
pca1 = PCA(n_components=2)
pca1=pca1.fit_transform(X_pca)
#we fit the pca model to the already transformed data X_pca which was transformed initially by pca

#function to perform kmeans clustering
#we apply kmeans clustering to the pca transformed data

def kmeans(data, k, nstart): #k=no. of clusters, nstart=the no. of times algorithm should be run

    #we randomly select initial clusters from the data without replacement
    np.random.seed(0)
    centers = data[np.random.choice(range(len(data)), k, replace=False)]

    for _ in range(nstart):   #runs this nstart times

        #computes euclidean dist. between data pts. and cluster center
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        #assigns each data pt. to the nearest cluster center based on euclidean dist.
        labels = np.argmin(distances, axis=1)
        #calculates new cluster centers as mean of data pts. assigned to each cluster
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        if np.all(centers == new_centers):
            break                       #breaks the loop if cluster centers dont change
        centers = new_centers

    wcss = 0     #: Initializes the Within-Cluster Sum of Squares (WCSS) to zero

    for i in range(k):   # Iterates over each cluster
        cluster_points = data[labels == i]
        wcss += np.sum(np.linalg.norm(cluster_points - centers[i], axis=1)**2)
        #computes sum of squared dist b/w each data pt in the cluster and the cluster centre and adds it to wcss

    return labels, wcss #Returns the final cluster labels and the WCSS value to the fn

#function to plot the elbow method
#we plot the elbow method to determine the optimal number of clusters

def plot_wcss(data, max_k, nstart):
    #max_k is the maximum no. of clusters to consider, 
    #nstart is the no. of times algorithm should be run

    #initialize an empty list to store wcss
    wcss_values = []
    
    for k in range(1, max_k + 1): #iterates over a range from 1 to kmax

        _, wcss = kmeans(data, k, nstart) #performs kmeans clustering for given k and gives its wcss
        wcss_values.append(wcss)           #we append the wcss value to wcss list
    
    #we plot all the wcss values against their respective k values and see the elbow point by eye
    plt.plot(range(1, max_k + 1), wcss_values, marker='o')  
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.show()

#plotting the elbow method
plot_wcss(pca1,20,25) #20 is the max value of k and 25 is no. of iterations for kmeans fn

#PLOTTING THE FINAL WCSS FOR DETERMINED K

c1,wcss1 = kmeans(pca1,2,25)
# 2 is the value of k found by eye from elbow method, 25 is no. of iterations of kmeans

#we create a scatter plot of the data pts in the reduced dimensional space obtained from pca
#plotting value of pc1 and pc2, c1 represents the colour of the cluster points obtained from the kmeans clustering
plt.scatter(pca1[:, 0], pca1[:, 1], c=c1)
plt.show()
print("The Within-Cluster Sum of Square is",wcss1)
#Prints the Within-Cluster Sum of Squares (WCSS) for the chosen number of clusters
