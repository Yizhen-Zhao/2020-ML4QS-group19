from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
import sys
import copy
import pandas as pd
import numpy as np

DATA_PATH = './intermediate_datafiles/'
dataset = pd.read_csv(DATA_PATH + 'chapter2_result.csv',index_col=0)
dataset.index = pd.to_datetime(dataset.index)

DataViz = VisualizeDataset()

milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

outlier_columns = ['acc_phone_x']

OutlierDist = DistanceBasedOutlierDetection()

#Run last two methods here and change parameters to get different figures
for col in outlier_columns:
	##distance-based approach， tried 0.11,0.99 and 0.50,0.99
    # dataset_outliers_sdb = OutlierDist.simple_distance_based(copy.deepcopy(dataset), [col], 'euclidean', 0.50, 0.99)
    # DataViz.plot_binary_outliers(dataset_outliers_sdb, col, 'simple_dist_outlier')

    #LOF approach, tried k=2 and k=9
    dataset = OutlierDist.local_outlier_factor(dataset, [col], 'euclidean', 9)
    DataViz.plot_dataset(dataset, [col, 'lof'], ['exact', 'exact'], ['line', 'points'])


