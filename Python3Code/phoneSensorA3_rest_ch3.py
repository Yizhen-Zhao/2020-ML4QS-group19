import sys
import copy
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters

# Set up the file names and locations.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = sys.argv[1] if len(sys.argv) > 1 else 'phoneSensorsA3_outliers_ch3.csv'
RESULT_FNAME = sys.argv[2] if len(sys.argv) > 2 else 'phoneSensorsA3_ch3_final.csv'
ORIG_DATASET_FNAME = sys.argv[3] if len(sys.argv) > 3 else 'phoneSensorsA3_ch2.csv'

try:
    dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

# We'll create an instance of our visualization class to plot the results.
DataViz = VisualizeDataset(__file__)

# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000


# impute_columns = ['acc_mobile_x', 'acc_mobile_y', 'acc_mobile_z', 'gyr_mobile_x','gyr_mobile_y','gyr_mobile_z','mag_mobile_x','mag_mobile_y','mag_mobile_z','prox_mobile_distance',
#                        'loc_mobile_latitude','loc_mobile_longitude','loc_mobile_height','loc_mobile_velocity','loc_mobile_direction','loc_mobile_horizontalAccuracy','loc_mobile_verticalAccuracy']
#

# Let us impute the missing values and plot an example.

MisVal = ImputationMissingValues()
imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'acc_mobile_x')
imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'acc_mobile_x')
imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'acc_mobile_x')
DataViz.plot_imputed_values(dataset, ['original', 'mean', 'interpolation'], 'acc_mobile_x', imputed_mean_dataset['acc_mobile_x'], imputed_interpolation_dataset['acc_mobile_x'])

# Now, let us carry out that operation over all columns except for the label.

for col in [c for c in dataset.columns if not 'label' in c]:
    dataset = MisVal.impute_mean(dataset, col)



# Using the result from Chapter 2, let us try the Kalman filter on the light_phone_lux attribute and study the result.

original_dataset = pd.read_csv(DATA_PATH / ORIG_DATASET_FNAME, index_col=0)
original_dataset.index = pd.to_datetime(original_dataset.index)
KalFilter = KalmanFilters()
kalman_dataset = KalFilter.apply_kalman_filter(original_dataset, 'acc_mobile_x')

DataViz.plot_imputed_values(kalman_dataset, ['original', 'kalman'], 'acc_mobile_x', kalman_dataset['acc_mobile_x_kalman'])
DataViz.plot_dataset(kalman_dataset, ['acc_mobile_x', 'acc_mobile_x_kalman'], ['exact','exact'], ['line', 'line'])


# Determine the PC's for all but our target columns (the labels and the heart rate)
# We simplify by ignoring both, we could also ignore one first, and apply a PC to the remainder.

PCA = PrincipalComponentAnalysis()
selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c))]
pc_values = PCA.determine_pc_explained_variance(dataset, selected_predictor_cols)

# Plot the variance explained.
DataViz.plot_xy(x=[range(1, len(selected_predictor_cols)+1)], y=[pc_values],
                xlabel='principal component number', ylabel='explained variance',
                ylim=[0,1], line_styles=['b-'])


# We select 6 as the best number of PC's as this explains most of the variance

n_pcs = 6

dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)

#And we visualize the result of the PC's

DataViz.plot_dataset(dataset, ['pca_', 'label'], ['like', 'like'], ['line', 'points'])

# And the overall final dataset:

DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'mag_', 'prox_', 'loc_', 'pca_', 'label'],
                     ['like', 'like', 'like', 'like', 'like', 'like', 'like'],
                     ['line', 'line', 'line', 'points', 'line', 'points', 'points'])






# Store the outcome.

dataset.to_csv(DATA_PATH / RESULT_FNAME)