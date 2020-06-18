##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.FeatureSelection import FeatureSelectionRegression
from util import util
from util.VisualizeDataset import VisualizeDataset

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter3_result_outliers_A3.csv'
RESULT_FNAME = 'HA-classification_result.csv'
EXPORT_TREE_PATH = Path('./figures/HA_classification2/')

# Next, we declare the parameters we'll use in the algorithms.
N_FORWARD_SELECTION = 12

try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset.index = pd.to_datetime(dataset.index)

# Let us create our visualization class again.
DataViz = VisualizeDataset(__file__)

# Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

# We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
# for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
# cases where we do not know the label.

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['hd'], 'dgdf', 0.7, filter=True, temporal=False)

print('Training set length is: ', len(train_X.index))
print('Test set length is: ', len(test_X.index))

# Select subsets of the features that we will consider:

basic_features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
# pca_features = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7']
# time_features = [name for name in dataset.columns if '_temp_' in name]
# freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
print('#basic features: ', len(basic_features))
# print('#PCA features: ', len(pca_features))
# print('#time features: ', len(time_features))
# print('#frequency features: ', len(freq_features))
cluster_features = ['cluster']
print('#cluster features: ', len(cluster_features))
# features_after_chapter_3 = list(set().union(basic_features, pca_features))
# features_after_chapter_4 = list(set().union(basic_features, pca_features))
features_after_chapter_5 = list(set().union(basic_features, cluster_features))


# First, let us consider the performance over a selection of features:

# fs = FeatureSelectionClassification()

# features, ordered_features, ordered_scores = fs.forward_selection(N_FORWARD_SELECTION,
#                                                                   train_X[basic_features], train_y)
# print(ordered_scores)
# print(ordered_features)

# DataViz.plot_xy(x=[range(1, N_FORWARD_SELECTION+1)], y=[ordered_scores],
#                 xlabel='number of features', ylabel='accuracy')
# print(features)

# # Based on the plot we select the top 10 features (note: slightly different compared to Python 2, we use
# # those feartures here).

selected_features = ['thal', 'oldpeak', 'exang', 'cp', 'chol']

# Let us first study the impact of regularization and model complexity: does regularization prevent overfitting?

learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()

reg_parameters = [0.0001, 0.001, 0.01, 0.1, 1, 10]
performance_training = []
performance_test = []

# We repeat the experiment a number of times to get a bit more robust data as the initialization of the NN is random.
N_REPEATS_NN = 20

# for reg_param in reg_parameters:
#     performance_tr = 0
#     performance_te = 0
#     for i in range(0, N_REPEATS_NN):
#         print("iteration %d" % i)

#         class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
#             train_X, train_y,
#             test_X, hidden_layer_sizes=(250, ), alpha=reg_param, max_iter=500,
#             gridsearch=False
#         )

#         performance_tr += eval.accuracy(train_y, class_train_y)
#         performance_te += eval.accuracy(test_y, class_test_y)
#     performance_training.append(performance_tr/N_REPEATS_NN)
#     performance_test.append(performance_te/N_REPEATS_NN)

# DataViz.plot_xy(x=[reg_parameters, reg_parameters], y=[performance_training, performance_test], method='semilogx',
#                 xlabel='regularization parameter value', ylabel='accuracy',
#                 names=['training', 'test'], line_styles=['r-', 'b:'])

# Second, let us consider the influence of certain parameter settings for the tree model. (very related to the
# regularization) and study the impact on performance.

# leaf_settings = [1,2,5,10]
# performance_training = []
# performance_test = []

# for no_points_leaf in leaf_settings:

#     class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
#         train_X[selected_features], train_y, test_X[selected_features], min_samples_leaf=no_points_leaf,
#         gridsearch=False, print_model_details=False)

#     performance_training.append(eval.accuracy(train_y, class_train_y))
#     performance_test.append(eval.accuracy(test_y, class_test_y))

# DataViz.plot_xy(x=[leaf_settings, leaf_settings], y=[performance_training, performance_test],
#                 xlabel='minimum number of points per leaf', ylabel='accuracy',
#                 names=['training', 'test'], line_styles=['r-', 'b:'])

# So yes, it is important :) Therefore we perform grid searches over the most important parameters, and do so by means
# of cross validation upon the training set.

possible_feature_sets = [basic_features, selected_features]
feature_names = ['initial set', 'Selected features']
N_KCV_REPEATS = 5

scores_over_all_algs = []

for i in range(0, len(possible_feature_sets)):
    print("feature set %d" % i)
    selected_train_X = train_X[possible_feature_sets[i]]
    selected_test_X = test_X[possible_feature_sets[i]]

    # First we run our non deterministic classifiers a number of times to average their score.

    performance_tr_nn = 0
    performance_tr_rf = 0
    performance_tr_svm = 0
    performance_te_nn = 0
    performance_te_rf = 0
    performance_te_svm = 0

    for repeat in range(0, N_KCV_REPEATS):
        print("repeat nr %d" % repeat)
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )
        print("finished ffnn ")
        
        performance_tr_nn += eval.accuracy(train_y, class_train_y)
        performance_te_nn += eval.accuracy(test_y, class_test_y)

        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )
        print("finished RF ")
        performance_tr_rf += eval.accuracy(train_y, class_train_y)
        performance_te_rf += eval.accuracy(test_y, class_test_y)

        # class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(
        #     selected_train_X, train_y, selected_test_X, gridsearch=True
        # )
        # print("finished svm ")
        # performance_tr_svm += eval.accuracy(train_y, class_train_y)
        # performance_te_svm += eval.accuracy(test_y, class_test_y)


    overall_performance_tr_nn = performance_tr_nn/N_KCV_REPEATS
    overall_performance_te_nn = performance_te_nn/N_KCV_REPEATS
    overall_performance_tr_rf = performance_tr_rf/N_KCV_REPEATS
    overall_performance_te_rf = performance_te_rf/N_KCV_REPEATS
    overall_performance_tr_svm = 0
    overall_performance_te_svm = 0
    # And we run our deterministic classifiers:

    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.k_nearest_neighbor(
        selected_train_X, train_y, selected_test_X, gridsearch=True
    )
    print("finished knn ")
    performance_tr_knn = eval.accuracy(train_y, class_train_y)
    performance_te_knn = eval.accuracy(test_y, class_test_y)

    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
        selected_train_X, train_y, selected_test_X, gridsearch=True
    )
    print("finished dt ")
    performance_tr_dt = eval.accuracy(train_y, class_train_y)
    performance_te_dt = eval.accuracy(test_y, class_test_y)

    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(
        selected_train_X, train_y, selected_test_X
    )
    print("finished NB ")
    performance_tr_nb = eval.accuracy(train_y, class_train_y)
    performance_te_nb = eval.accuracy(test_y, class_test_y)

    scores_with_sd = util.print_table_row_performances(feature_names[i], len(selected_train_X.index), len(selected_test_X.index), [
                                                                                                (overall_performance_tr_nn, overall_performance_te_nn),
                                                                                                (overall_performance_tr_rf, overall_performance_te_rf),
                                                                                                (overall_performance_tr_svm, overall_performance_te_svm),
                                                                                                (performance_tr_knn, performance_te_knn),
                                                                                                (performance_tr_dt, performance_te_dt),
                                                                                                (performance_tr_nb, performance_te_nb)])
    scores_over_all_algs.append(scores_with_sd)

DataViz.plot_performances_classification(['NN', 'RF', 'SVM', 'KNN', 'DT', 'NB'], feature_names, scores_over_all_algs)

# And we study two promising ones in more detail. First, let us consider the decision tree, which works best with the
# selected features.

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(train_X[selected_features], train_y, test_X[selected_features],
                                                                                           gridsearch=True,
                                                                                           print_model_details=True)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
    train_X[selected_features], train_y, test_X[selected_features],
    gridsearch=True, print_model_details=True)

test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)

DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)
