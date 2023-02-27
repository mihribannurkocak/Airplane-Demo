import requests
import json
import csv
import time
import numpy as np
import os
import math
from sklearn.cluster import KMeans
from sklearn import preprocessing


#TO REQUEST DATA FROM API
def getDataFromApi(min_lat,min_lon,max_lat,max_lon):
    api_url = 'https://opensky-network.org/api/states/all?'+'lamin=' + \
        str(min_lat)+'&lomin='+str(min_lon)+'&lamax=' + \
        str(max_lat)+'&lomax='+str(max_lon)

    responses = requests.get(api_url)
    while responses is None:
        print("Data Cannot be Fetched!")
        responses = requests.get(api_url)
    responses = responses.json()
    all_required_data = []
    for plane in responses['states']:
        if(plane[5] != None and plane[6] != None and plane[10] != None): #long, lat, true_track
            all_required_data.append(plane)
    return all_required_data

def createPlanesDictionary(all_plane_data):
    plane_features = np.zeros((len(all_plane_data),4)) #features for classification
    planes_dicts = []
    i = 0
    print("Number of planes: ", len(all_plane_data))
    for plane in all_plane_data:
        plane_features[i][0] = plane[5] #latitude
        plane_features[i][1] = plane[6] #longtitude
        plane_features[i][2] = plane[10] #true_track

        plane_dict = {}
        plane_dict["icao24"] = plane[0]
        plane_dict["callsign"]= plane[1]
        plane_dict["origin_country"]= plane[2]
        plane_dict["long"] = plane[5]
        plane_dict["lat"] = plane[6]
        plane_dict["geo_altitude"]= plane[13]
        plane_dict["velocity"]= plane[9]
        plane_dict["true_track"]= plane[10]
        planes_dicts.append(plane_dict)
        i+=1
    return planes_dicts, plane_features

#SCORING METHOD TO FIND BEST NUMBER OF CLUSTERS(K)
def bicScore(X: np.ndarray, labels: np.array):
  """
  BIC score for the goodness of fit of clusters.
  This Python function is translated from the Golang implementation by the author of the paper. 
  The original code is available here: https://github.com/bobhancock/goxmeans/blob/a78e909e374c6f97ddd04a239658c7c5b7365e5c/km.go#L778
  """
    
  n_points = len(labels)
  n_clusters = len(set(labels))
  n_dimensions = X.shape[1]
  n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1

  loglikelihood = 0
  for label_name in set(labels):
    X_cluster = X[labels == label_name]
    n_points_cluster = len(X_cluster)
    centroid = np.mean(X_cluster, axis=0)
    if(len(X_cluster) == 1):
      variance = 0.0000000001 #error factor to avoid loglikelihood is infinity
    else: 
      variance = np.sum((X_cluster - centroid) ** 2) / (len(X_cluster) - 1)

    loglikelihood += \
      n_points_cluster * np.log(n_points_cluster) \
      - n_points_cluster * np.log(n_points) \
      - n_points_cluster * n_dimensions / 2 * np.log(2 * math.pi * variance) \
      - (n_points_cluster - 1) / 2
    
  bic = loglikelihood - (n_parameters / 2) * np.log(n_points)
  return bic


#TO SCALE FEATURES BEFORE CLASSIFICATION
def scaleFeatures(plane_features):
    scaler = preprocessing.MinMaxScaler() #Min Max Scaler
    plane_features_scaled = scaler.fit_transform(plane_features)
    return plane_features_scaled

#TO CLASSIFY AIRPLANES
def classifyPlanes(plane_features):
    plane_features_scaled = scaleFeatures(plane_features)

    #bic scoring method
    k = 2
    kmeans = KMeans(n_clusters=2,random_state=42)
    kmeans.fit(plane_features_scaled)
    score = bicScore(plane_features_scaled, kmeans.labels_) #worst score with 2 clusters

    for candidate_k in range(3,11): 
        kmeans = KMeans(n_clusters=candidate_k,random_state=42)
        kmeans.fit(plane_features_scaled)
        local_score = bicScore(plane_features_scaled, kmeans.labels_)
        if(local_score >= score):
            k = candidate_k
            score = local_score
        
    kmeans = KMeans(n_clusters=k, random_state=42)
    print("Number of clusters(k):",k)
    kmeans.fit(plane_features_scaled)
    return kmeans.labels_,k


#TO AVOID CONSTANT COLOR CHANGES FOR EACH CLASS---------------------------------------
def find_next_class(prev_classes,i_plane,plane_classes):
    while i_plane < len(plane_classes) and plane_classes[i_plane] in prev_classes:
        i_plane+=1
    next_class = plane_classes[i_plane]
    return next_class, i_plane+1


def assign_classes(plane_classes,planes_dict,k):
    i_class = 0
    i_plane = 0
    discovered_classes = []
    while i_class < k:
        c, i_plane = find_next_class(discovered_classes,i_plane,plane_classes)
        discovered_classes.append(c)
        i_class+=1
    
    i = 0
    planes_dict_with_classes = []
    for plane in planes_dict:
        c_plane = discovered_classes.index(plane_classes[i])
        plane['class'] = c_plane
        planes_dict_with_classes.append(plane)
        i+=1
    return planes_dict_with_classes
#-------------------------------------------------------------------------------------

#MAIN FUNCTION
def main():
    """
    min_lat = -0.936  # north atlantic coordinates
    min_lon = -98.0539 
    max_lat = 68.6387
    max_lon = 12.0059 
    """

    """
    min_lat = 19.50139 # US coordinates
    min_lon = -161.75583
    max_lat = 64.85694
    max_lon = -68.01197
    """

    min_lat = 35.9025  # turkey coordinates
    min_lon = 25.90902 
    max_lat = 42.02683
    max_lon = 44.5742

   
    out_col_names = ['icao24', 'callsign', 'origin_country', 'long', 'lat', 'geo_altitude', 'velocity','true_track','class']

    #Main Loop
    while True:
        all_required_data = getDataFromApi(min_lat,min_lon,max_lat,max_lon)
        all_required_data.sort()
        planes_dict, plane_features= createPlanesDictionary(all_required_data)
        plane_classes, k = classifyPlanes(plane_features)
        planes_dict = assign_classes(plane_classes,planes_dict,k)
        with open('planes.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = out_col_names)
            writer.writeheader()
            writer.writerows(planes_dict)
        time.sleep(10) #Wait for 10 seconds

if __name__=="__main__":
    main()
