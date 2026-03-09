# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. choose the number of clusters(K)

2. Randomly initialise k centroids

3. assign each data point to the nearest centroid

4. recalculate the centroids


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: 
RegisterNumber:  
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mail_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
print(data.head())
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)


data['Cluster'] = y_kmeans

print("\nClustered Data:")
print(data.head())


plt.figure()
plt.scatter(X[y_kmeans == 0]['Annual Income (k$)'], 
            X[y_kmeans == 0]['Spending Score (1-100)'], label='Cluster 0')

plt.scatter(X[y_kmeans == 1]['Annual Income (k$)'], 
            X[y_kmeans == 1]['Spending Score (1-100)'], label='Cluster 1')

plt.scatter(X[y_kmeans == 2]['Annual Income (k$)'], 
            X[y_kmeans == 2]['Spending Score (1-100)'], label='Cluster 2')

plt.scatter(X[y_kmeans == 3]['Annual Income (k$)'], 
            X[y_kmeans == 3]['Spending Score (1-100)'], label='Cluster 3')

plt.scatter(X[y_kmeans == 4]['Annual Income (k$)'], 
            X[y_kmeans == 4]['Spending Score (1-100)'], label='Cluster 4')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], 
            s=200, label='Centroids')

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
```

## Output:
<img width="834" height="194" alt="image" src="https://github.com/user-attachments/assets/3f3e842a-8783-4f34-b62d-5eb523ee5ef8" />
<img width="877" height="361" alt="image" src="https://github.com/user-attachments/assets/e82f3a5f-700a-4d14-a9ed-19fe1138a34d" />
<img width="940" height="656" alt="image" src="https://github.com/user-attachments/assets/8c68f978-013c-496f-8cb6-8d3d6e8715d8" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
