import sys
sys.path.insert(1, './imports')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def average_squared_loss_from_log(y_pred, y_test):
    return (sum((np.exp(y_pred) - np.exp(y_test))**2))/len(y_pred)

def average_squared_loss(y_pred, y_test):
    return (sum(((y_pred) - (y_test))**2))/len(y_pred)

def loss_for_n_estimators(n_estimators, X_train, X_test, y_train, y_test):
    min_squared_loss = np.Inf
    losses = []
    for i,estimators in enumerate(n_estimators):
        rfr = RandomForestRegressor(n_estimators = estimators)
        rfr.fit(X_train, y_train)
        y_pred = rfr.predict(X_test)
        temp_loss = average_squared_loss_from_log(y_pred, y_test)
        losses.append(temp_loss)
        if temp_loss < min_squared_loss:
            min_squared_loss = temp_loss
            min_estimator = estimators
    return losses

def greedy_feature_removal(X_train, y_train, X_test, y_test):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    importances = rfc.feature_importances_
    imp_series = pd.Series(data=importances, index=X_train.columns)
    test_accs = []
    imp_series.sort_values(ascending=True, inplace=True)
    for i, feature in enumerate(imp_series.index):
        temp_X_train = X_train.loc[:,imp_series.index].iloc[:,i:]
        temp_X_test = X_test.loc[:,imp_series.index].iloc[:,i:]
        rfc = RandomForestClassifier()
        rfc.fit(temp_X_train, y_train)
        y_pred = rfc.predict(temp_X_test)
        test_accs.append(sum(y_pred == y_test)/len(y_pred))
    return test_accs

def greedy_feature_removal_oob(X, y):
    rfc = RandomForestClassifier()
    rfc.fit(X, y)
    importances = rfc.feature_importances_
    imp_series = pd.Series(data=importances, index=X.columns)
    oobs = {}
    imp_series.sort_values(ascending=True, inplace=True)
    for i, feature in enumerate(imp_series.index):
        temp_X = X.loc[:,imp_series.index].iloc[:,i:]
        rfc = RandomForestClassifier(warm_start=True, oob_score=True)
        rfc.fit(temp_X, y)
        oobs[i]= rfc.oob_score_
    return oobs

def max_depth_oobs(X, y):
    depths = np.arange(1, 100, 2)
    oobs = {}
    for i, d in enumerate(depths):
        rfc = RandomForestClassifier(warm_start=True, oob_score=True, max_depth=d)
        rfc.fit(X, y)
        oobs[i]= rfc.oob_score_
    return oobs

def average_oob(X, X2, y):
    """
        returns for two different sets of variables the accuracies as variables are removed
    """
    cum_dict_lf = {}
    cum_dict_hf = {}
    for i in np.arange(0, 10):
        if i == 0:
            cum_dict_lf = greedy_feature_removal_oob(X2, y)
            cum_dict_hf = greedy_feature_removal_oob(X, y)
        else:
            temp_dict_lf = greedy_feature_removal_oob(X2, y)
            temp_dict_hf = greedy_feature_removal_oob(X, y)
            for key in cum_dict_lf.keys():
                cum_dict_lf[key] += temp_dict_lf[key]
                cum_dict_hf[key] += temp_dict_hf[key]
    for key in cum_dict_lf.keys():
        cum_dict_lf[key] = cum_dict_lf[key]/10
        cum_dict_hf[key] = cum_dict_hf[key]/10
    return (pd.Series(cum_dict_lf), pd.Series(cum_dict_hf))

def select_kernel_kcross(X, X_with_area, y, splits=10, repeats=5):
    """
        Returns the area burnt and accuracy using stratified cross validation
        [X: the whole data]
        [y: the target variable]
        [X_with_area: the dataset including the area burnt]
        [Splits: Number of subsets to split the data in for the stratified cross validation]
        [Repeats: Number of times to repeat the whole cross validation]
    """
    kernels = ['linear', 'rbf', 'poly']
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats,random_state=3558)
    areas_burnt = {}
    test_accuracies = {}
    for kernel in kernels:
        print(kernel)
        n = 0
        cum_accuracies = 0
        cum_area_burnt = 0
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            rfc = SVC(kernel=kernel)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            cum_accuracies+= accuracy
            cum_area_burnt += area_burnt(X_with_area.iloc[test_index], y_pred, y_test)
            n += 1
        areas_burnt[kernel] = (cum_area_burnt)/n
        test_accuracies[kernel] = (cum_accuracies)/n
    return areas_burnt, test_accuracies

def area_burnt(X_test_with_area, y_pred, y_test):
    """
        Returns the sum over the false positives of the area burnt in that fire.
        [X_test_with_area: The data containing the area burnt]
        [y_pred: The predictions done by the algorithm]
        [y_test: the actual values]
    """
    false_negatives = (y_pred == 0) & (y_test == 1)
    return sum(np.exp(X_test_with_area.loc[false_negatives]["area"]))

def select_c_kcross(X, X_with_area, y,  splits=10, repeats=5, gamma = 1.6768329368110066e-05):
     """
        Select C using stratified cross validation
        [X: the whole data]
        [y: the target variable]
        [X_with_area: the dataset including the area burnt]
        [min_g: The minimum gamma to be scored in logarithmic scale]
        [max_g: The maximum gamma to be scored in logarithmic scale]
        [Splits: Number of subsets to split the data in for the stratified cross validation]
        [Repeats: Number of times to repeat the whole cross validation]
        [gamma: The value of gamma]
    """
    cs = np.logspace(-10, 2, 50)
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats,random_state=3558)
    areas_burnt = {}
    test_accuracies = {}
    for c in cs:
        n = 0
        cum_accuracies = 0
        cum_area_burnt = 0
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            rfc = SVC(kernel = "rbf", gamma = gamma, C=c)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            cum_accuracies+= accuracy
            cum_area_burnt += area_burnt(X_with_area.iloc[test_index], y_pred, y_test)
            n += 1
        areas_burnt[c] = (cum_area_burnt)/n
        test_accuracies[c] = (cum_accuracies)/n
    return areas_burnt, test_accuracies

def select_gamma_kcross(X, X_with_area, y, min_g, max_g, splits=10, repeats=5):
    """
        Select gamma using stratified cross validation
        [X: the whole data]
        [y: the target variable]
        [X_with_area: the dataset including the area burnt]
        [min_g: The minimum gamma to be scored in logarithmic scale]
        [max_g: The maximum gamma to be scored in logarithmic scale]
        [Splits: Number of subsets to split the data in for the stratified cross validation]
        [Repeats: Number of times to repeat the whole cross validation]
    """
    gammas = np.logspace(min_g, max_g, 50)
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats,random_state=3558)
    areas_burnt = {}
    test_accuracies = {}
    for gamma in gammas:
        n = 0
        cum_accuracies = 0
        cum_area_burnt = 0
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            rfc = SVC(kernel = "rbf", gamma=gamma)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            cum_accuracies+= accuracy
            cum_area_burnt += area_burnt(X_with_area.iloc[test_index], y_pred, y_test)
            n += 1
        areas_burnt[gamma] = (cum_area_burnt)/n
        test_accuracies[gamma] = (cum_accuracies)/n
    return areas_burnt, test_accuracies

def select_c_kcross_refinement(X, X_with_area, y, min_c =1.5, max_c=2.5, splits=10, repeats=5):
    """
        Select c in a smaller range using stratified cross validation
        [X: the whole data]
        [y: the target variable]
        [X_with_area: the dataset including the area burnt]
        [min_c: The minimum c to be scored in logarithmic scale]
        [max_c: The maximum c to be scored in logarithmic scale]
        [Splits: Number of subsets to split the data in for the stratified cross validation]
        [Repeats: Number of times to repeat the whole cross validation]
    """
    cs = np.linspace(1.5, 2.5, 50)
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats,random_state=3558)
    areas_burnt = {}
    test_accuracies = {}
    for c in cs:
        n = 0
        cum_accuracies = 0
        cum_area_burnt = 0
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            rfc = SVC(kernel = "rbf", gamma = 2.2229964825261955e-05, C=c)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            cum_accuracies+= accuracy
            cum_area_burnt += area_burnt(X_with_area.iloc[test_index], y_pred, y_test)
            n += 1
        areas_burnt[c] = (cum_area_burnt)/n
        test_accuracies[c] = (cum_accuracies)/n
    return areas_burnt, test_accuracies

def crossval_feature_removal(X, y, X_with_area, gamma, C, splits=10, repeats=5):
    """
        Orders features importance using an initial random forest and removes them greedily, evaluating each removal using a stratified cross-validation approach, checking the accuracy and the area burnt.
        [X: the whole data]
        [y: the target variable]
        [X_with_area: the dataset including the area burnt]
        [gamma: the value of gamma for the SVM]
        [C: The value of C for the SVM]
        [Splits: Number of subsets to split the data in for the stratified cross validation]
        [Repeats: Number of times to repeat the whole cross validation]
    """
    rfc = RandomForestClassifier()
    rfc.fit(X, y)
    importances = rfc.feature_importances_
    imp_series = pd.Series(data=importances, index=X.columns)
    oobs = {}
    imp_series.sort_values(ascending=True, inplace=True)
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=5,random_state=3558)
    areas_burnt = {}
    test_accuracies = {}
    for var, feature in enumerate(imp_series.index):
        n = 0
        cum_accuracies = 0
        cum_area_burnt = 0
        for train_index, test_index in rskf.split(X, y):
            if var > 0:
                X_temp = X.copy().iloc[:,0:-var]
            else:
                X_temp = X.copy()
            X_train, X_test = X_temp.iloc[train_index], X_temp.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            rfc = SVC(kernel = "rbf", gamma=gamma, C=C)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            cum_accuracies+= accuracy
            cum_area_burnt += area_burnt(X_with_area.iloc[test_index], y_pred, y_test)
            n += 1
        areas_burnt[var] = (cum_area_burnt)/n
        test_accuracies[var] = (cum_accuracies)/n
    return imp_series, areas_burnt, test_accuracies