# Libraries
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV, \
    PredefinedSplit
from skrebate import ReliefF
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif, RFECV, SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score, recall_score
from scipy.stats import entropy
import json
from collections import defaultdict

############################################ Datasets & Data preparation ############################################

# Dataset number 1- Breast Cancer Wisconsin (Diagnostic)
bc_data = load_breast_cancer()
bc_dataset = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)
bc_dataset['target'] = bc_data.target
print(bc_dataset.head())

# Dataset number 2 - Mobile Price Classification
mobile_data = pd.read_csv('Mobile_Price.csv')
mobile_dataset = mobile_data.drop('price_range', axis=1)
mobile_dataset['target'] = mobile_data['price_range']
print(mobile_dataset.head())

# Dataset number 3 - Iris Dataset
iris_data = load_iris()
iris_dataset = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_dataset['target'] = iris_data.target
print(iris_dataset.head())

# Dataset number 4 - Wine Dataset
wine_data = load_wine()
wine_dataset = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_dataset['target'] = wine_data.target
print(wine_dataset.head())

# Dataset number 5 - mushrooms Dataset
mushrooms_data = pd.read_csv('mushrooms.csv')

# Dataset number 6 - Zoo Dataset
zoo_data = pd.read_csv('zoo.csv')
zoo_data = zoo_data.drop('animal_name', axis=1)
zoo_data.rename(columns={'class_type': 'target'}, inplace=True)


# Train and Test split for all datasets
bc_train, bc_test = train_test_split(bc_dataset, test_size=0.2, stratify=bc_dataset['target'], random_state=42)
mobile_train, mobile_test = train_test_split(mobile_dataset, test_size=0.2, stratify=mobile_dataset['target'], random_state=42)
iris_train, iris_test = train_test_split(iris_dataset, test_size=0.2, stratify=iris_dataset['target'], random_state=42)
wine_train, wine_test = train_test_split(wine_dataset, test_size=0.2, stratify=wine_dataset['target'], random_state=42)
# Splitting before creating dummies
mushrooms_train_raw, mushrooms_test_raw = train_test_split(mushrooms_data, test_size=0.2, stratify=mushrooms_data['class'], random_state=42)
zoo_train_raw, zoo_test_raw = train_test_split(zoo_data, test_size=0.2, stratify=zoo_data['target'], random_state=42)

# get_dummies after split for mushrooms & zoo
# Mushrooms
X_train_mush = pd.get_dummies(mushrooms_train_raw.drop('class', axis=1))
y_train_mush = mushrooms_train_raw['class'].map({'e': 0, 'p': 1})
X_test_mush = pd.get_dummies(mushrooms_test_raw.drop('class', axis=1))
y_test_mush = mushrooms_test_raw['class'].map({'e': 0, 'p': 1})

X_test_mush = X_test_mush.reindex(columns=X_train_mush.columns, fill_value=0)

mushrooms_train = X_train_mush.copy()
mushrooms_train['target'] = y_train_mush
mushrooms_test = X_test_mush.copy()
mushrooms_test['target'] = y_test_mush
print(mushrooms_train.head())

# Zoo
X_train_zoo = pd.get_dummies(zoo_train_raw.drop('target', axis=1), columns=['legs'])
y_train_zoo = zoo_train_raw['target']
X_test_zoo = pd.get_dummies(zoo_test_raw.drop('target', axis=1), columns=['legs'])
y_test_zoo = zoo_test_raw['target']

X_test_zoo = X_test_zoo.reindex(columns=X_train_zoo.columns, fill_value=0)

zoo_train = X_train_zoo.copy()
zoo_train['target'] = y_train_zoo
zoo_test = X_test_zoo.copy()
zoo_test['target'] = y_test_zoo
print(zoo_train.head())


########################################### Feature Selection Methods ############################################

# Dictionary for saving the selected features
feature_selection_results = defaultdict(dict)

# Function that updates the dictionary
def add_selected_features(results_dict, dataset_name, method_name, feature_list):
    results_dict[dataset_name][method_name] = list(feature_list)

########################################### Feature Selection - Filter methods

# SU - Symmetrical Uncertainty
def symmetrical_uncertainty_feature_selection(data, target_column='target',
                                               method_name='Symmetrical Uncertainty',
                                               dataset_name='Dataset'):

    start_time = time.time()
    print(f"\n--- Method: {method_name} | Dataset: {dataset_name} ---")

    X = data.drop(columns=target_column).copy()
    y = data[target_column].copy()

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    mi = mutual_info_classif(X_scaled, y, discrete_features='auto', random_state=42)
    H_x = np.array([entropy(np.histogram(X_scaled[col], bins=10)[0], base=2) for col in X_scaled.columns])
    H_y = entropy(np.histogram(y, bins=len(np.unique(y)))[0], base=2)

    su_scores = 2 * mi / (H_x + H_y)
    su_df = pd.DataFrame({'Feature': X_scaled.columns, 'SU Score': su_scores})
    su_df.sort_values(by='SU Score', ascending=False, inplace=True)
    print(su_df)

    # Use top-k = log2(n)
    top_k = int(np.ceil(np.log2(len(X_scaled.columns))))
    selected_features = su_df['Feature'].head(top_k).values
    data_selected = data[selected_features]

    end_time = time.time()
    runtime_seconds = end_time - start_time

    print(f"Selected top {top_k} features (log2-based)")
    print(f"Optimal number of features: {len(selected_features)}")
    print(f"Selected features: {list(selected_features)}")
    print(f"Feature selection runtime: {runtime_seconds:.2f} seconds")

    return data_selected, selected_features, runtime_seconds

# Use on all datasets
su_bc_df, features_bc_su, su_bc_time = symmetrical_uncertainty_feature_selection(bc_train, target_column='target', dataset_name='Breast Cancer')
su_mobile_df, features_mobile_su, su_mobile_time = symmetrical_uncertainty_feature_selection(mobile_train, target_column='target', dataset_name='Mobile')
su_iris_df, features_iris_su, su_iris_time = symmetrical_uncertainty_feature_selection(iris_train, target_column='target', dataset_name='Iris')
su_wine_df, features_wine_su, su_wine_time = symmetrical_uncertainty_feature_selection(wine_train, target_column='target', dataset_name='Wine')
su_mush_df, features_mush_su, su_mush_time = symmetrical_uncertainty_feature_selection(mushrooms_train, target_column='target', dataset_name='mushrooms')
su_zoo_df, features_zoo_su, su_zoo_time = symmetrical_uncertainty_feature_selection(zoo_train, target_column='target', dataset_name='Zoo')


# Add selected features to dictionary
add_selected_features(feature_selection_results, 'Breast Cancer', 'SU', features_bc_su)
add_selected_features(feature_selection_results, 'Mobile', 'SU', features_mobile_su)
add_selected_features(feature_selection_results, 'Iris', 'SU', features_iris_su)
add_selected_features(feature_selection_results, 'Wine', 'SU', features_wine_su)
add_selected_features(feature_selection_results, 'mushrooms', 'SU', features_mush_su)
add_selected_features(feature_selection_results, 'Zoo', 'SU', features_zoo_su)


# MI - Mutual Information
def mutual_info_feature_selection(data, target_column='target',
                                  method_name='Mutual Information',
                                  dataset_name='Dataset'):

    start_time = time.time()
    print(f"\n--- Method: {method_name} | Dataset: {dataset_name} ---")

    X = data.drop(columns=target_column).copy()
    y = data[target_column].copy()

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    mi_scores = mutual_info_classif(X_scaled, y, discrete_features='auto', random_state=0)
    mi_df = pd.DataFrame({'Feature': X_scaled.columns, 'MI Score': mi_scores})
    mi_df.sort_values(by='MI Score', ascending=False, inplace=True)
    print(mi_df)

    # Select top_k based on log2(n)
    top_k = int(np.ceil(np.log2(len(X_scaled.columns))))
    selected_features = mi_df['Feature'].head(top_k).values
    data_selected = data[selected_features]

    end_time = time.time()
    runtime_seconds = end_time - start_time

    print(f"Selected top {top_k} features (log2-based)")
    print(f"Optimal number of features: {len(selected_features)}")
    print(f"Selected features: {selected_features}")
    print(f"Feature selection runtime: {runtime_seconds:.2f} seconds")

    return data_selected, selected_features, runtime_seconds


# Use on all datasets
mi_bc_df, features_bc_mi, mi_bc_time = mutual_info_feature_selection(bc_train, target_column='target', dataset_name='Breast Cancer')
mi_mobile_df, features_mobile_mi, mi_mobile_time = mutual_info_feature_selection(mobile_train, target_column='target', dataset_name='Mobile')
mi_iris_df, features_iris_mi, mi_iris_time = mutual_info_feature_selection(iris_train, target_column='target', dataset_name='Iris')
mi_wine_df, features_wine_mi, mi_wine_time = mutual_info_feature_selection(wine_train, target_column='target', dataset_name='Wine')
mi_mush_df, features_mush_mi, mi_mush_time = mutual_info_feature_selection(mushrooms_train, target_column='target', dataset_name='mushrooms')
mi_zoo_df, features_zoo_mi, mi_zoo_time = mutual_info_feature_selection(zoo_train, target_column='target', dataset_name='Zoo')

# Add selected features to dictionary
add_selected_features(feature_selection_results, 'Breast Cancer', 'MI', features_bc_mi)
add_selected_features(feature_selection_results, 'Mobile', 'MI', features_mobile_mi)
add_selected_features(feature_selection_results, 'Iris', 'MI', features_iris_mi)
add_selected_features(feature_selection_results, 'Wine', 'MI', features_wine_mi)
add_selected_features(feature_selection_results, 'mushrooms', 'MI', features_mush_mi)
add_selected_features(feature_selection_results, 'Zoo', 'MI', features_zoo_mi)


# ReliefF
def relief_feature_selection(data, target_column='target',
                             n_neighbors=50,
                             method_name='ReliefF',
                             dataset_name='Dataset'):

    start_time = time.time()
    print(f"\n--- Method: {method_name} | Dataset: {dataset_name} ---")

    X = data.drop(columns=target_column).copy()
    y = data[target_column].copy()

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    relief = ReliefF(n_neighbors=n_neighbors)
    relief.fit(X_scaled.values, y)

    relief_scores = relief.feature_importances_
    relief_df = pd.DataFrame({'Feature': X_scaled.columns, 'ReliefF Score': relief_scores})
    relief_df.sort_values(by='ReliefF Score', ascending=False, inplace=True)
    print(relief_df)

    # Select top_k based on log2(n)
    top_k = int(np.ceil(np.log2(len(X_scaled.columns))))
    selected_features = relief_df['Feature'].head(top_k).values
    data_selected = data[selected_features]

    end_time = time.time()
    runtime_seconds = end_time - start_time

    print(f"Selected top {top_k} features (log2-based)")
    print(f"Optimal number of features: {len(selected_features)}")
    print(f"Selected features: {selected_features}")
    print(f"Feature selection runtime: {runtime_seconds:.2f} seconds")

    return data_selected, selected_features, runtime_seconds


# Use on all datasets
rf_bc_df, features_bc_rf, rf_bc_time = relief_feature_selection(bc_train, target_column='target', dataset_name='Breast Cancer')
rf_mobile_df, features_mobile_rf, rf_mobile_time = relief_feature_selection(mobile_train, target_column='target', dataset_name='Mobile')
rf_iris_df, features_iris_rf, rf_iris_time = relief_feature_selection(iris_train, target_column='target', dataset_name='Iris')
rf_wine_df, features_wine_rf, rf_wine_time = relief_feature_selection(wine_train, target_column='target', dataset_name='Wine')
rf_mush_df, features_mush_rf, rf_mush_time = relief_feature_selection(mushrooms_train, target_column='target', dataset_name='mushrooms')
rf_zoo_df, features_zoo_rf, rf_zoo_time = relief_feature_selection(zoo_train, target_column='target', dataset_name='Zoo')

# Add selected features to dictionary
add_selected_features(feature_selection_results, 'Breast Cancer', 'rf', features_bc_rf)
add_selected_features(feature_selection_results, 'Mobile', 'rf', features_mobile_rf)
add_selected_features(feature_selection_results, 'Iris', 'rf', features_iris_rf)
add_selected_features(feature_selection_results, 'Wine', 'rf', features_wine_rf)
add_selected_features(feature_selection_results, 'mushrooms', 'rf', features_mush_rf)
add_selected_features(feature_selection_results, 'Zoo', 'rf', features_zoo_rf)


# Chi2 - For categorical datasets only
def chi_square_feature_selection(data, target_column='target', top_k=None, p_threshold=0.05, method_name='Chi-Square', dataset_name='Dataset'):

    start_time = time.time()

    print(f"\n--- Method: {method_name} | Dataset: {dataset_name} ---")

    X = data.drop(columns=target_column).copy()
    y = data[target_column].copy()
    chi_scores, p_values = chi2(X, y)
    chi_df = pd.DataFrame({'Feature': X.columns, 'Chi2 Score': chi_scores, 'p-value': p_values})
    chi_df.sort_values(by='Chi2 Score', ascending=False, inplace=True)
    print(chi_df)
    if top_k is None:
        selected_features = chi_df[chi_df['p-value'] < p_threshold]['Feature'].values
    else:
        selected_features = chi_df['Feature'].head(top_k).values
    data_selected = data[selected_features]

    end_time = time.time()
    runtime_seconds = end_time - start_time

    print(f"Optimal number of features: {len(selected_features)}")
    print(f"Selected features: {selected_features}")
    print(f"Feature selection runtime: {runtime_seconds:.2f} seconds")

    return data_selected, selected_features, runtime_seconds



# Use on all categorical datasets
cs_mush_df, features_mush_cs, cs_mush_time = chi_square_feature_selection(mushrooms_train, target_column='target', dataset_name='mushrooms')
cs_zoo_df, features_zoo_cs, cs_zoo_time = chi_square_feature_selection(zoo_train, target_column='target', dataset_name='Zoo')

# Add selected features to dictionary
add_selected_features(feature_selection_results, 'mushrooms', 'cs', features_mush_cs)
add_selected_features(feature_selection_results, 'Zoo', 'cs', features_zoo_cs)


# ANOVA F-test - For continuous datasets only
def anova_f_feature_selection(data, target_column='target', top_k=None, p_threshold=0.05, method_name='ANOVA F-test', dataset_name='Dataset'):

    start_time = time.time()
    print(f"\n--- Method: {method_name} | Dataset: {dataset_name} ---")

    X = data.drop(columns=target_column).copy()
    y = data[target_column].copy()

    # Compute F-scores and p-values
    f_scores, p_values = f_classif(X, y)
    f_df = pd.DataFrame({'Feature': X.columns, 'F Score': f_scores, 'p-value': p_values})
    f_df.sort_values(by='F Score', ascending=False, inplace=True)
    print(f_df)

    # Feature selection
    if top_k is not None:
        selected_features = f_df['Feature'].head(top_k).values
    else:
        selected_features = f_df[f_df['p-value'] < p_threshold]['Feature'].values

    data_selected = data[selected_features]

    end_time = time.time()
    runtime_seconds = end_time - start_time

    print(f"Optimal number of features: {len(selected_features)}")
    print(f"Selected features: {selected_features}")
    print(f"Feature selection runtime: {runtime_seconds:.2f} seconds")

    return data_selected, selected_features, runtime_seconds


# Use on all numerical datasets
anova_bc_df, features_bc_anova, anova_bc_time = anova_f_feature_selection(bc_train, target_column='target', dataset_name='Breast Cancer')
anova_iris_df, features_iris_anova, anova_iris_time = anova_f_feature_selection(iris_train, target_column='target', dataset_name='Iris')
anova_wine_df, features_wine_anova, anova_wine_time = anova_f_feature_selection(wine_train, target_column='target', dataset_name='Wine')

# Add selected features to dictionary
add_selected_features(feature_selection_results, 'Breast Cancer', 'anova', features_bc_anova)
add_selected_features(feature_selection_results, 'Iris', 'anova', features_iris_anova)
add_selected_features(feature_selection_results, 'Wine', 'anova', features_wine_anova)

############################################ Feature Selection - Wrapper methods

# RFECV - Recursive Feature Elimination with Cross-Validation
def rfe_feature_selection(data, target_column='target', estimator=RandomForestClassifier(random_state=42), cv_folds=5, scoring='f1_macro', method_name='RFE', dataset_name='Dataset'):

    start_time = time.time()

    print(f"\n--- Method: {method_name} | Dataset: {dataset_name} ---")

    X = data.drop(columns=target_column).copy()
    y = data[target_column].copy()

    rfecv = RFECV(estimator=estimator,
                  step=1,
                  cv=StratifiedKFold(cv_folds,shuffle=True, random_state=42),
                  scoring=scoring,
                  n_jobs=1)

    rfecv.fit(X, y)

    selected_features = X.columns[rfecv.support_]
    data_selected = data[selected_features]

    end_time = time.time()
    runtime_seconds = end_time - start_time

    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"Selected features: {selected_features.tolist()}")
    print(f"Feature selection runtime: {runtime_seconds:.2f} seconds")

    return data_selected, selected_features, runtime_seconds



# Use on all datasets
rfe_bc_df, features_bc_rfe, rfe_bc_time = rfe_feature_selection(bc_train, target_column='target', dataset_name='Breast Cancer')
rfe_mobile_df, features_mobile_rfe, rfe_mobile_time = rfe_feature_selection(mobile_train, target_column='target', dataset_name='Mobile')
rfe_iris_df, features_iris_rfe, rfe_iris_time = rfe_feature_selection(iris_train, target_column='target', dataset_name='Iris')
rfe_wine_df, features_wine_rfe, rfe_wine_time = rfe_feature_selection(wine_train, target_column='target', dataset_name='Wine')
rfe_mush_df, features_mush_rfe, rfe_mush_time = rfe_feature_selection(mushrooms_train, target_column='target', dataset_name='mushrooms')
rfe_zoo_df, features_zoo_rfe, rfe_zoo_time = rfe_feature_selection(zoo_train, target_column='target',cv_folds=3, dataset_name='Zoo')

# Add selected features to dictionary
add_selected_features(feature_selection_results, 'Breast Cancer', 'rfe', features_bc_rfe)
add_selected_features(feature_selection_results, 'Mobile', 'rfe', features_mobile_rfe)
add_selected_features(feature_selection_results, 'Iris', 'rfe', features_iris_rfe)
add_selected_features(feature_selection_results, 'Wine', 'rfe', features_wine_rfe)
add_selected_features(feature_selection_results, 'mushrooms', 'rfe', features_mush_rfe)
add_selected_features(feature_selection_results, 'Zoo', 'rfe', features_zoo_rfe)


# SFS - Sequential Feature Selection
def sfs_feature_selection(data, target_column='target',
                          estimator=RandomForestClassifier(random_state=42),
                          method_name='SFS - Forward Selection',
                          dataset_name='Dataset'):

    start_time = time.time()
    print(f"\n--- Method: {method_name} | Dataset: {dataset_name} ---")

    X = data.drop(columns=target_column).copy()
    y = data[target_column].copy()

    sfs = SequentialFeatureSelector(estimator=estimator,
                                     n_features_to_select='auto',
                                     direction='forward',
                                     scoring='f1_macro',
                                     cv=None,
                                     n_jobs=1)

    sfs.fit(X, y)

    selected_features = X.columns[sfs.get_support()]
    data_selected = data[selected_features]

    end_time = time.time()
    runtime_seconds = end_time - start_time

    print(f"Optimal number of features: {len(selected_features)}")
    print(f"Selected features: {selected_features.tolist()}")
    print(f"Feature selection runtime: {runtime_seconds:.2f} seconds")

    return data_selected, selected_features, runtime_seconds


# Use on all datasets
sfs_bc_df, features_bc_sfs, sfs_bc_time = sfs_feature_selection(bc_train, target_column='target', dataset_name='Breast Cancer')
sfs_mobile_df, features_mobile_sfs, sfs_mobile_time = sfs_feature_selection(mobile_train, target_column='target', dataset_name='Mobile')
sfs_iris_df, features_iris_sfs, sfs_iris_time = sfs_feature_selection(iris_train, target_column='target', dataset_name='Iris')
sfs_wine_df, features_wine_sfs, sfs_wine_time = sfs_feature_selection(wine_train, target_column='target', dataset_name='Wine')
sfs_mush_df, features_mush_sfs, sfs_mush_time = sfs_feature_selection(mushrooms_train, target_column='target', dataset_name='mushrooms')
sfs_zoo_df, features_zoo_sfs, sfs_zoo_time = sfs_feature_selection(zoo_train, target_column='target', dataset_name='Zoo')

# Add selected features to dictionary
add_selected_features(feature_selection_results, 'Breast Cancer', 'sfs', features_bc_sfs)
add_selected_features(feature_selection_results, 'Mobile', 'sfs', features_mobile_sfs)
add_selected_features(feature_selection_results, 'Iris', 'sfs', features_iris_sfs)
add_selected_features(feature_selection_results, 'Wine', 'sfs', features_wine_sfs)
add_selected_features(feature_selection_results, 'mushrooms', 'sfs', features_mush_sfs)
add_selected_features(feature_selection_results, 'Zoo', 'sfs', features_zoo_sfs)


# Backward Elimination Feature Selection
def backward_elimination_feature_selection(data, target_column='target',
                                           estimator=RandomForestClassifier(random_state=42),
                                           method_name='Backward Elimination',
                                           dataset_name='Dataset'):

    start_time = time.time()
    print(f"\n--- Method: {method_name} | Dataset: {dataset_name} ---")

    X = data.drop(columns=target_column)
    y = data[target_column]

    sfs = SequentialFeatureSelector(estimator,
                                    n_features_to_select='auto',
                                    direction='backward',
                                    scoring='f1_macro',
                                    cv=None,
                                    n_jobs=1)

    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()].tolist()

    end_time = time.time()
    runtime_seconds = end_time - start_time

    print(f"Optimal number of features: {len(selected_features)}")
    print(f"Selected features: {selected_features}")
    print(f"Feature selection runtime: {runtime_seconds:.2f} seconds")

    data_selected = data[selected_features + [target_column]]

    return data_selected, selected_features, runtime_seconds



# Use on all datasets
be_bc_df, features_bc_be, be_bc_time = backward_elimination_feature_selection(bc_train, target_column='target', dataset_name='Breast Cancer')
be_mobile_df, features_mobile_be, be_mobile_time = backward_elimination_feature_selection(mobile_train, target_column='target', dataset_name='Mobile')
be_iris_df, features_iris_be, be_iris_time = backward_elimination_feature_selection(iris_train, target_column='target', dataset_name='Iris')
be_wine_df, features_wine_be, be_wine_time = backward_elimination_feature_selection(wine_train, target_column='target', dataset_name='Wine')
be_mush_df, features_mush_be, be_mush_time = backward_elimination_feature_selection(mushrooms_train, target_column='target', dataset_name='mushrooms')
be_zoo_df, features_zoo_be, be_zoo_time = backward_elimination_feature_selection(zoo_train, target_column='target', dataset_name='Zoo')

# Add selected features to dictionary
add_selected_features(feature_selection_results, 'Breast Cancer', 'be', features_bc_be)
add_selected_features(feature_selection_results, 'Mobile', 'be', features_mobile_be)
add_selected_features(feature_selection_results, 'Iris', 'be', features_iris_be)
add_selected_features(feature_selection_results, 'Wine', 'be', features_wine_be)
add_selected_features(feature_selection_results, 'mushrooms', 'be', features_mush_be)
add_selected_features(feature_selection_results, 'Zoo', 'be', features_zoo_be)

############################################ Feature Selection - Hybrid methods

# First approach - Filter then wrapper selection
# Mutual Information + RFECV
def filter_then_wrapper_selection(data, target_column='target',
                                  estimator=RandomForestClassifier(random_state=42),
                                  mi_threshold=0.05,
                                  cv_folds=5,
                                  scoring='f1_macro',
                                  method_name='Hybrid: Filter (MI) + Wrapper (RFE)',
                                  dataset_name='Dataset'):
    start_time = time.time()

    print(f"\n--- Method: {method_name} | Dataset: {dataset_name} ---")

    # Separate features and target
    X = data.drop(columns=target_column).copy()
    y = data[target_column].copy()

    # Normalize features (for MI)
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Step 1: Filter - Mutual Information
    mi_scores = mutual_info_classif(X_scaled, y, discrete_features='auto', random_state=0)
    mi_df = pd.DataFrame({'Feature': X_scaled.columns, 'MI Score': mi_scores})
    selected_by_mi = mi_df[mi_df['MI Score'] > mi_threshold]['Feature'].values

    if len(selected_by_mi) == 0:
        print("No features passed the MI threshold.")
        return None, [], 0

    if len(selected_by_mi) == 1:
        print("Only one feature passed the MI threshold. Skipping wrapper step.")
        selected_features = selected_by_mi
        data_selected = data[list(selected_features)]

        end_time = time.time()
        runtime_seconds = end_time - start_time

        print(f"Selected feature: {selected_features[0]}")
        print(f"Feature selection runtime: {runtime_seconds:.2f} seconds")

        return data_selected, selected_features, runtime_seconds


    # Step 2: Wrapper - RFECV
    X_filtered = X[selected_by_mi]
    rfecv = RFECV(estimator=estimator,
                  step=1,
                  cv=StratifiedKFold(n_splits=cv_folds,shuffle=True, random_state=42),
                  scoring=scoring,
                  n_jobs=1)

    rfecv.fit(X_filtered, y)
    selected_features = X_filtered.columns[rfecv.support_]
    data_selected = data[selected_features]

    end_time = time.time()
    runtime_seconds = end_time - start_time

    print(f"Features after MI filtering: {len(selected_by_mi)}")
    print(f"Final selected features after RFE: {len(selected_features)}")
    print(f"Selected features: {selected_features.tolist()}")
    print(f"Feature selection runtime: {runtime_seconds:.2f} seconds")

    return data_selected, selected_features, runtime_seconds

# # Use on all datasets
hybrid1_bc_df, features_bc_hybrid1, hybrid1_bc_time = filter_then_wrapper_selection(bc_train, target_column='target', dataset_name='Breast Cancer')
hybrid1_mobile_df, features_mobile_hybrid1, hybrid1_mobile_time = filter_then_wrapper_selection(mobile_train, target_column='target', dataset_name='Mobile')
hybrid1_iris_df, features_iris_hybrid1, hybrid1_iris_time = filter_then_wrapper_selection(iris_train, target_column='target', dataset_name='Iris')
hybrid1_wine_df, features_wine_hybrid1, hybrid1_wine_time = filter_then_wrapper_selection(wine_train, target_column='target', dataset_name='Wine')
hybrid1_mush_df, features_mush_hybrid1, hybrid1_mush_time = filter_then_wrapper_selection(mushrooms_train, target_column='target', dataset_name='mushrooms')
hybrid1_zoo_df, features_zoo_hybrid1, hybrid1_zoo_time = filter_then_wrapper_selection(zoo_train, target_column='target',cv_folds=3, dataset_name='Zoo')

# Add selected features to dictionary
add_selected_features(feature_selection_results, 'Breast Cancer', 'hybrid1', features_bc_hybrid1)
add_selected_features(feature_selection_results, 'Mobile', 'hybrid1', features_mobile_hybrid1)
add_selected_features(feature_selection_results, 'Iris', 'hybrid1', features_iris_hybrid1)
add_selected_features(feature_selection_results, 'Wine', 'hybrid1', features_wine_hybrid1)
add_selected_features(feature_selection_results, 'mushrooms', 'hybrid1', features_mush_hybrid1)
add_selected_features(feature_selection_results, 'Zoo', 'hybrid1', features_zoo_hybrid1)


# Second approach - Weighted Hybrid: Filter + Embedded
# Mutual Information + Random forest importance
def weighted_hybrid_feature_selection(data, target_column='target',
                                      filter_weight=0.5,
                                      model_weight=0.5,
                                      top_k=None,
                                      threshold=0.1,
                                      method_name='Hybrid, Weighted: Filter (MI) + Embedded (Random Forest)',
                                      dataset_name='Dataset',
                                      n_jobs=1,
                                      cv_folds=5):

    start_time = time.time()
    print(f"\n--- Method: {method_name} | Dataset: {dataset_name} ---")

    X = data.drop(columns=target_column).copy()
    y = data[target_column].copy()

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Step 1: Mutual Information
    mi_scores = mutual_info_classif(X_scaled, y, discrete_features='auto', random_state=42)
    mi_scores_scaled = MinMaxScaler().fit_transform(mi_scores.reshape(-1, 1)).flatten()

    # Step 2: Random Forest Importances with CV
    rf = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
    rf_importances_all = []

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, y_train = X_scaled.iloc[train_idx], y.iloc[train_idx]
        model = clone(rf)
        model.fit(X_train, y_train)
        rf_importances_all.append(model.feature_importances_)

    rf_importances_avg = np.mean(rf_importances_all, axis=0)
    rf_scores_scaled = MinMaxScaler().fit_transform(rf_importances_avg.reshape(-1, 1)).flatten()

    # Step 3: Weighted Score
    final_scores = filter_weight * mi_scores_scaled + model_weight * rf_scores_scaled
    scores_df = pd.DataFrame({'Feature': X.columns, 'Weighted Score': final_scores})
    scores_df.sort_values(by='Weighted Score', ascending=False, inplace=True)
    print(scores_df)

    # Select Features
    if top_k is not None:
        selected_features = scores_df['Feature'].head(top_k).values
    else:
        selected_features = scores_df[scores_df['Weighted Score'] > threshold]['Feature'].values

    data_selected = data[selected_features]

    end_time = time.time()
    runtime_seconds = end_time - start_time

    print(f"Optimal number of features: {len(selected_features)}")
    print(f"Selected features: {list(selected_features)}")
    print(f"Feature selection runtime: {runtime_seconds:.2f} seconds")

    return data_selected, selected_features, runtime_seconds

# # Use on all datasets
hybrid2_bc_df, features_bc_hybrid2, hybrid2_bc_time = weighted_hybrid_feature_selection(bc_train, target_column='target', dataset_name='Breast Cancer')
hybrid2_mobile_df, features_mobile_hybrid2, hybrid2_mobile_time = weighted_hybrid_feature_selection(mobile_train, target_column='target', dataset_name='Mobile')
hybrid2_iris_df, features_iris_hybrid2, hybrid2_iris_time = weighted_hybrid_feature_selection(iris_train, target_column='target', dataset_name='Iris')
hybrid2_wine_df, features_wine_hybrid2, hybrid2_wine_time = weighted_hybrid_feature_selection(wine_train, target_column='target', dataset_name='Wine')
hybrid2_mush_df, features_mush_hybrid2, hybrid2_mush_time = weighted_hybrid_feature_selection(mushrooms_train, target_column='target', dataset_name='mushrooms')
hybrid2_zoo_df, features_zoo_hybrid2, hybrid2_zoo_time = weighted_hybrid_feature_selection(zoo_train, target_column='target',cv_folds=3, dataset_name='Zoo')

# Add selected features to dictionary
add_selected_features(feature_selection_results, 'Breast Cancer', 'hybrid2', features_bc_hybrid2)
add_selected_features(feature_selection_results, 'Mobile', 'hybrid2', features_mobile_hybrid2)
add_selected_features(feature_selection_results, 'Iris', 'hybrid2', features_iris_hybrid2)
add_selected_features(feature_selection_results, 'Wine', 'hybrid2', features_wine_hybrid2)
add_selected_features(feature_selection_results, 'mushrooms', 'hybrid2', features_mush_hybrid2)
add_selected_features(feature_selection_results, 'Zoo', 'hybrid2', features_zoo_hybrid2)


############################################ Modeling - Random Forest ############################################

# Write to a json file the selected features
with open('feature_selection_results.json', 'w') as f:
    json.dump(feature_selection_results, f, indent=2)




# Open the json file of the selected features (after the first run of the above)
with open('feature_selection_results.json', 'r') as f:
    feature_selection_results = json.load(f)

os.makedirs("plots", exist_ok=True)

# Function that Returns train/test sets based on selected features
def prepare_data_for_model(dataset_df_train, dataset_df_test, dataset_name, method_name, selected_features_dict):
    selected_features = selected_features_dict[dataset_name][method_name]
    X_train = dataset_df_train[selected_features]
    y_train = dataset_df_train['target']
    X_test = dataset_df_test[selected_features]
    y_test = dataset_df_test['target']
    return X_train, X_test, y_train, y_test


# Function to print scores
def evaluate_model(model, X_train, y_train, X_test, y_test, dataset_name, fs_method):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "Dataset - FS Method": [f"{dataset_name} - {fs_method}"] * 3,
        "Metric": ["F1-Macro", "Recall-Macro","Precision-Macro"],
        "Train Set": [
            f1_score(y_train, y_train_pred, average='macro'),
            recall_score(y_train, y_train_pred, average='macro'),
            precision_score(y_train, y_train_pred, average='macro'),
        ],
        "Test Set": [
            f1_score(y_test, y_test_pred, average='macro'),
            recall_score(y_test, y_test_pred, average='macro'),
            precision_score(y_test, y_test_pred, average='macro'),
        ]
    }

    metrics_df = pd.DataFrame(metrics)
    return metrics_df


def run_rf_with_random_search(X_train, y_train, X_test, y_test,
                              dataset_name, feature_selection_method,
                              n_iter=60, cv_folds=10):

    print(f"Dataset: {dataset_name} | Feature Selection: {feature_selection_method}")

    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': np.arange(8, 15, 1),
        'criterion': ['gini', 'entropy'],
        'max_features': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='f1_macro',
        cv=cv_folds,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    best_model.fit(X_train, y_train)

    print("After RandomizedSearch, the best Random Forest model is:", best_model)
    print("Best parameters:", search.best_params_)

    results_df = evaluate_model(best_model, X_train, y_train, X_test, y_test,
                                dataset_name=dataset_name,
                                fs_method=feature_selection_method)
    print(results_df)

    # Feature importances
    importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title(f"Feature Importance: {dataset_name} ({feature_selection_method})")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save plot to 'plots' folder
    filename = f"plots/feature_importance_{dataset_name}_{feature_selection_method}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    return best_model, search.best_params_, results_df, feature_importance_df



############################################ Breast Cancer


datasets = {
    'Breast Cancer': (bc_train, bc_test),
    'Mobile': (mobile_train, mobile_test),
    'Iris': (iris_train, iris_test),
    'Wine': (wine_train, wine_test),
    'mushrooms': (mushrooms_train, mushrooms_test),
    'Zoo': (zoo_train, zoo_test)
}


def run_all_models(datasets, feature_selection_results):
    all_results = []

    for dataset_name, (train_df, test_df) in datasets.items():
        if dataset_name not in feature_selection_results:
            print(f"{dataset_name} not found in feature selection results. Skipping.")
            continue

        fs_methods = feature_selection_results[dataset_name]

        for fs_method, selected_features in fs_methods.items():
            print(f"Running {dataset_name} with feature selection: {fs_method}")

            X_train, X_test, y_train, y_test = prepare_data_for_model(
                train_df, test_df, dataset_name, fs_method, feature_selection_results
            )

            best_model, best_params, results_df, feature_importance_df = run_rf_with_random_search(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                dataset_name=dataset_name,
                feature_selection_method=fs_method
            )

            results_df['Dataset'] = dataset_name
            results_df['FS Method'] = fs_method
            results_df['Best Params'] = [best_params] * len(results_df)

            all_results.append(results_df)

    # Combine all into one DataFrame
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        return final_df
    else:
        print("No results to return.")
        return pd.DataFrame()







final_results_df = run_all_models(datasets, feature_selection_results)
