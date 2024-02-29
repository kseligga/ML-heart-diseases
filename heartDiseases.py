import numpy as np
import warnings
import pandas as pd
from sklearn.preprocessing import scale
from pandas import DataFrame
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster import hierarchy
import cluster
import kmedoids
import gower


raw_df = pd.read_csv("heart_disease_patients.csv")

warnings.filterwarnings('ignore')


raw_df = raw_df.drop(raw_df.columns[0], axis=1)

print(raw_df.dtypes)
summart = raw_df.describe()

import copy
df = copy.deepcopy(raw_df)
df

df['sex'] = np.where(df['sex']==1, 'val1',
                   np.where(df['sex']==0, 'val0', 'val'))
df['cp'] = np.where(df['cp']==1, 'val1',
                    np.where(df['cp']==2, 'val2',
                    np.where(df['cp']==3, 'val3', 
                    np.where(df['cp']==4, 'val4', 'val'))))
df['fbs'] = np.where(df['fbs']==1, 'val1',
                    np.where(df['fbs']==0, 'val0', 'val'))
df['restecg'] = np.where(df['restecg']==0, 'val0',
                    np.where(df['restecg']==1, 'val1',
                    np.where(df['restecg']==2, 'val2', 'val')))
df['exang'] = np.where(df['exang']==1, 'val1',
                    np.where(df['exang']==0, 'val0', 'val'))
df['slope'] = np.where(df['slope']==3, 'val3',
                    np.where(df['slope']==1, 'val1',
                    np.where(df['slope']==2, 'val2', 'val')))
df

print(df.dtypes)
df.describe()

distance_matrix = gower.gower_matrix(df)
raw_distance_matrix = gower.gower_matrix(raw_df)
print(distance_matrix)
print()
print(raw_distance_matrix)

print((distance_matrix < 0.1).sum()/303)

xd = (distance_matrix < 0.2).sum(axis=1)

(xd > 5).sum()

from scipy.cluster import hierarchy
Z = hierarchy.linkage(distance_matrix, method='ward')
plt.figure(figsize=(10,5), dpi=200)
hierarchy.dendrogram(Z)
plt.show()


agglModelGower = AgglomerativeClustering(n_clusters=4, linkage='ward')
agglModelGower.fit(distance_matrix)
aggl_y = agglModelGower.labels_
aggl_y

clustered_df = copy.deepcopy(raw_df)
clustered_df['cluster'] = aggl_y

clustered_df


dis1 = clustered_df[clustered_df['cluster'] == 1]
dis0 = clustered_df[clustered_df['cluster'] == 0]
dis2 = clustered_df[clustered_df['cluster'] == 2]
dis3 = clustered_df[clustered_df['cluster'] == 3]

plt.hist(dis1['cp'])
plt.title('Type of chest pain')
plt.show()
plt.hist(dis1['sex'])
plt.title('Gender')
plt.show()
plt.hist(dis1['exang'])
plt.title('Exercise induced angina')
plt.show()
plt.hist(dis0['age'], alpha = 0.5)
plt.hist(dis1['age'], alpha = 0.5)
plt.hist(dis2['age'], alpha = 0.5)
plt.hist(dis3['age'], alpha = 0.5)
plt.show()

plt.hist(dis0['chol'], alpha = 0.5)
plt.hist(dis1['chol'], alpha = 0.5)
plt.hist(dis2['chol'], alpha = 0.5)
plt.hist(dis3['chol'], alpha = 0.5)
plt.show()

plt.hist(dis0['oldpeak'])
plt.title('ST segment depression')
plt.show()
plt.hist(dis0['sex'])
plt.title('Gender')
plt.show()
plt.hist(dis0['thalach'])
plt.title('Max heart rate')
plt.show()

plt.hist(dis2['oldpeak'])
plt.title('ST segment depression')
plt.show()
plt.hist(dis2['sex'])
plt.title('Gender')
plt.show()
plt.hist(dis2['thalach'])
plt.title('Max heart rate')
plt.show()
plt.hist(dis2['slope'])
plt.title('ST segment slope')
plt.show()

plt.hist(dis3['slope'])
plt.title('ST segment slope')
plt.show()
plt.hist(dis3['oldpeak'])
plt.title('ST segment depression')
plt.show()
plt.hist(dis3['trestbps'])
plt.title('Resting blood pressure')
plt.show()
plt.hist(dis3['fbs'])
plt.title('High fasting blood sugar')
plt.show()