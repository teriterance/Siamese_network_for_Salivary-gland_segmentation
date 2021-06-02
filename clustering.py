import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

def hieracjiclaClustering(data, feature_1, feature_2):
    im_shape = data.shape
    print(im_shape)
    X = np.array(list(zip(feature_1.flatten(), feature_2.flatten())))
    thresh = 1.5
    clusters = hcluster.fclusterdata(X, thresh, criterion='distance')
    img_out = np.reshape(clusters, im_shape)
    print(clusters)
    print(im_shape)
    plt.imshow(img_out)
    plt.show()
    return clusters

def clustering_KNN(img, feature_1, feature_2):
    im_shape = img.shape
    print(im_shape)
    #linearisation des image
    X = np.array(list(zip(feature_1.flatten(), feature_2.flatten())))
    print(X.shape)
    #dendrogram = sch.dendrogram(sch.linkage(X[0:20000], method='ward'))
    #hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    #y_hc = hc.fit_predict(X[0:20000])
    #print(y_hc)
    #print(y_hc.shape)
    kmeans = KMeans(n_clusters=6)
    y_km = kmeans.fit_predict(X)
    print(y_km)
    img_clust = np.reshape(y_km, im_shape)
    return img_clust

def clustering_DBSCAN(img, feature_1, feature_2):
    im_shape = img.shape
    print(im_shape)
    #linearisation des image
    X = np.array(list(zip(feature_1.flatten(), feature_2.flatten())))
    print(X.shape)
    model = DBSCAN(eps=0.30, min_samples=9)
    yhat = model.fit_predict(X[0:150000])
    print(yhat)
    img_clust = np.reshape(yhat, im_shape)
    plt.imshow(img_clust)
    plt.show()