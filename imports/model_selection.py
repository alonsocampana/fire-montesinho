import sys
sys.path.insert(1, './imports')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import skopt
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import forest_minimize

def add_feature(df, col1, col2):
    df = df.copy()
    new_name = col1 + " * " + col2
    map_names = {"col_name":new_name}
    return (df.assign(col_name = df.loc[:,col1] * df.loc[:,col2])).rename(map_names, axis=1)

def average_squared_loss_from_log(y_pred, y_test):
    """
        From a set of predictions, find the loss and transforms the area back to the original scale in hectareas
    """
    return (sum((np.exp(y_pred) - np.exp(y_test))**2))/len(y_pred)

def average_squared_loss(y_pred, y_test):
    return (sum(((y_pred) - (y_test))**2))/len(y_pred)

def average_absolute_loss(y_pred, y_test):
    return sum(abs(((y_pred) - (y_test))))/len(y_pred)

def loss_for_n_estimators(n_estimators, X_train, X_test, y_train, y_test):
    """
        Returns an array containing the losses in function of the number of estimators
    """
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
    """
        Removes features greedily and returns the accuracy score
    """
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
    """
        Removes features greedily and returns the out of bag score
    """
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
    """
        Returns out of bag score for different max depths
    """
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

def select_c_kcross(X, X_with_area, y,  splits=10, repeats=5, gamma = 1.6768329368110066e-05, kernel = "rbf"):
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
            rfc = SVC(kernel = kernel, gamma = gamma, C=c)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            cum_accuracies+= accuracy
            cum_area_burnt += area_burnt(X_with_area.iloc[test_index], y_pred, y_test)
            n += 1
        areas_burnt[c] = (cum_area_burnt)/n
        test_accuracies[c] = (cum_accuracies)/n
    return areas_burnt, test_accuracies

def select_gamma_kcross(X, X_with_area, y, min_g, max_g, splits=10, repeats=5, kernel = "rbf"):
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
            rfc = SVC(kernel = kernel, gamma=gamma)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            cum_accuracies+= accuracy
            cum_area_burnt += area_burnt(X_with_area.iloc[test_index], y_pred, y_test)
            n += 1
        areas_burnt[gamma] = (cum_area_burnt)/n
        test_accuracies[gamma] = (cum_accuracies)/n
    return areas_burnt, test_accuracies

def select_c_kcross_refinement(X, X_with_area, y, min_c =1.5, max_c=2.5, splits=10, repeats=5, kernel = "rbf"):
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
            rfc = SVC(kernel = kernel, gamma = 2.2229964825261955e-05, C=c)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            cum_accuracies+= accuracy
            cum_area_burnt += area_burnt(X_with_area.iloc[test_index], y_pred, y_test)
            n += 1
        areas_burnt[c] = (cum_area_burnt)/n
        test_accuracies[c] = (cum_accuracies)/n
    return areas_burnt, test_accuracies

def crossval_feature_removal(X, y, X_with_area, gamma, C, splits=10, repeats=5, kernel = "rbf"):
    """
        Orders features importance using an initial random forest and removes them greedily, evaluating each removal using a stratified cross-validation approach, checking the accuracy and the area burnt.
        [X: the whole data]
        [y: the target variable]
        [X_with_area: the dataset including the area burnt]
        [gamma: the value of gamma for the SVM]
        [C: The value of C for the SVM]
        [splits: Number of subsets to split the data in for the stratified cross validation]
        [repeats: Number of times to repeat the whole cross validation]
    """
    rfc = RandomForestClassifier()
    rfc.fit(X, y)
    importances = rfc.feature_importances_
    imp_series = pd.Series(data=importances, index=X.columns)
    oobs = {}
    imp_series.sort_values(ascending=True, inplace=True)
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats,random_state=3558)
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
            rfc = SVC(kernel = kernel, gamma=gamma, C=C)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            cum_accuracies+= accuracy
            cum_area_burnt += area_burnt(X_with_area.iloc[test_index], y_pred, y_test)
            n += 1
        areas_burnt[var] = (cum_area_burnt)/n
        test_accuracies[var] = (cum_accuracies)/n
    return imp_series, areas_burnt, test_accuracies

def SVM_hyperpar_skopt(X, y, min_c = 1e-6, max_c = 100.0, kernels= ['rbf'], max_degree=2, min_g = 1e-6, max_g = 100.0):
    """
        Explores the model space and returns the model with the optimal accuracy.
        [X: the data]
        [y: the target variable]
        [min_c: C lower bound]
        [max_c: C upper bound] 
        [kernels: array of kernels to be tested]
        [max_degree: max degree of the polynomial kernel]
        [min_g = gamma lower bound]
        [max_g = gamma upper bound]
    """
    search_space = list()
    search_space.append(Real(min_c, max_c, 'log-uniform', name='C'))
    search_space.append(Categorical(kernels, name='kernel'))
    search_space.append(Integer(1, max_degree, name='degree'))
    search_space.append(Real(min_g, max_g, 'log-uniform', name='gamma'))
    @use_named_args(search_space)
    def evaluate_model(**params):
        # configure the model with specific hyperparameters
        model = SVC()
        model.set_params(**params)
        # define test harness
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=3558)
        # calculate 5-fold cross validation
        result = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='accuracy')
        # calculate the mean of the scores
        estimate = np.mean(result)
        # convert from a maximizing score to a minimizing score
        return 1.0 - estimate
    result = gp_minimize(evaluate_model, search_space)
    return result

def crossval_ridge(X, y, degree, alpha):
    """
        Uses k-fold crossval to estimate the loss associated to a given degree of a polynomial feature map, and alpha regularization value.
    """
    X_temp = X.copy()
    featurizer = PolynomialFeatures(degree=degree)
    featurizer.fit(X)
    X_temp = featurizer.transform(X_temp)
    rskf = KFold(n_splits=5, shuffle=True,random_state=3558)
    losses = []
    n=0
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X_temp[train_index], X_temp[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = Ridge(alpha=alpha)
        n+=1
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        losses.append(average_squared_loss_from_log(y_pred, y_test))
    return sum(losses)/n

def hyperpar_grid_ridge(X, y, degrees = [1, 2, 3], min_alpha = -1, max_alpha = 2):
    """
        Uses grid search for estimating the loss associated to different combinations of degrees of the feature map and alpha regularization parameters
    """
    degrees = degrees
    alphas = np.logspace(min_alpha, max_alpha, 20)
    losses = np.zeros([len(degrees), len(alphas)])
    for i, deg in enumerate(degrees):
        for j, alpha in enumerate(alphas):
            losses[i, j] = crossval_ridge(X, y, deg, alpha)
    return pd.DataFrame(losses, index= degrees, columns = alphas)

def crossval_lasso(X, y, degree, alpha):
    """
        Uses k-fold crossval to estimate the loss associated to a given degree of a polynomial feature map, and alpha regularization value.
    """
    X = X.copy()
    featurizer = PolynomialFeatures(degree=degree)
    featurizer.fit(X)
    X = featurizer.transform(X)
    rskf = KFold(n_splits=5, shuffle=True,random_state=3558)
    losses = []
    n=0
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = Lasso(alpha=alpha)
        n+=1
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        losses.append(average_squared_loss_from_log(y_pred, y_test))
    return sum(losses)/n

def hyperpar_grid_lasso(X, y, degrees = [1, 2, 3], min_alpha = -1, max_alpha = 2):
    """
        Uses grid search for estimating the loss associated to different combinations of degrees of the feature map and alpha regularization parameters
    """
    degrees = degrees
    alphas = np.logspace(min_alpha, max_alpha, 20)
    losses = np.zeros([len(degrees), len(alphas)])
    for i, deg in enumerate(degrees):
        for j, alpha in enumerate(alphas):
            losses[i, j] = crossval_lasso(X, y, deg, alpha)
    return pd.DataFrame(losses, index= degrees, columns = alphas)

def gbr_score(X, y, degree, learning_rate, n_estimators):
    """
        For a given data and combination of parameters, results the squared loss resulting from k-fold cross validation
    """
    poly = PolynomialFeatures(degree=degree)
    poly.fit(X)
    X_temp = poly.transform(X)
    rskf = KFold(n_splits=5, shuffle=True,random_state=3558)
    losses = []
    n=0
    for train_index, test_index in rskf.split(X, y):
        n += 1
        X_train, X_test = X_temp[train_index], X_temp[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = GradientBoostingRegressor(learning_rate = learning_rate, n_estimators = n_estimators)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        losses.append(average_squared_loss_from_log(y_pred, y_test))
    return sum(losses)/n

def hyper_opt_gbr(X, y):
    """
        Iterates over a set of hyperparameters and returns the combination resulting in the lowest loss
    """
    degrees, learning_rates, n_estimatorss = [1, 2, 3], np.linspace(0.05, 0.6, 5), [20, 50, 100]
    min_loss = 100000
    for deg in degrees:
        for lr in learning_rates:
            for n_estimator in n_estimatorss:
                temp_loss = gbr_score(X, y, deg, lr, n_estimator)
                if temp_loss < min_loss:
                    min_loss = temp_loss
                    min_deg = deg
                    min_lr = lr
                    min_n_estimator = n_estimator
    return {"loss":min_loss, "deg": min_deg, "lr":min_lr, "n_estimators":min_n_estimator}

def hyper_opt_gbr_2(X, y):
    """
        Iterates over a set of hyperparameters and returns the combination resulting in the lowest loss
    """
    degrees, learning_rates, n_estimatorss = [1, 2, 3], np.linspace(0.05, 0.6, 5), [20, 50, 100]
    min_loss = 100000
    for deg in degrees:
        for lr in learning_rates:
            for n_estimator in n_estimatorss:
                temp_loss = gbr_score_2(X, y, deg, lr, n_estimator)
                if temp_loss < min_loss:
                    min_loss = temp_loss
                    min_deg = deg
                    min_lr = lr
                    min_n_estimator = n_estimator
    return {"loss":min_loss, "deg": min_deg, "lr":min_lr, "n_estimators":min_n_estimator}

def gbr_score_2(X, y, degree, learning_rate, n_estimators):
    """
        For a given data and combination of parameters, results the squared loss resulting from k-fold cross validation
    """
    poly = PolynomialFeatures(degree=degree)
    poly.fit(X)
    X_temp = poly.transform(X)
    rskf = KFold(n_splits=5, shuffle=True,random_state=3558)
    losses = []
    n=0
    for train_index, test_index in rskf.split(X, y):
        n += 1
        X_train, X_test = X_temp[train_index], X_temp[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = GradientBoostingRegressor(learning_rate = learning_rate, n_estimators = n_estimators)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        losses.append(average_absolute_loss(y_pred, y_test))
    return sum(losses)/n

def hyper_opt_rfr(X, y):
    """
        Iterates over a set of hyperparameters and returns the combination resulting in the lowest loss
    """
    degrees, criterion, n_estimatorss = [1, 2], ['mse', 'mae'], [20, 50, 100]
    min_loss = 100000
    for deg in degrees:
        for cr in criterion:
            for n_estimator in n_estimatorss:
                temp_loss = rfr_score(X, y, deg, cr, n_estimator)
                if temp_loss < min_loss:
                    min_loss = temp_loss
                    min_deg = deg
                    min_cr = cr
                    min_n_estimator = n_estimator
    return {"loss":min_loss, "deg": min_deg, "lr":min_cr, "n_estimators":min_n_estimator}

def rfr_score(X, y, degree, crit, n_estimators):
    """
        For a given data and combination of parameters, results the squared loss resulting from k-fold cross validation
    """
    poly = PolynomialFeatures(degree=degree)
    poly.fit(X)
    X_temp = poly.transform(X)
    rskf = KFold(n_splits=5, shuffle=True,random_state=3558)
    losses = []
    n=0
    for train_index, test_index in rskf.split(X, y):
        n += 1
        X_train, X_test = X_temp[train_index], X_temp[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = RandomForestRegressor(criterion=crit, n_estimators = n_estimators)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        losses.append(average_absolute_loss(y_pred, y_test))
    return sum(losses)/n

def greedy_addition(X, y, features, lr = 0.05, n_estimators = 100):
    """
        Given a list of features sorted by decreasing promisingness, they are added and the average loss of the model is computed.
        [X: the data]
        [y: The target variable]
        [features: A list containing pairs of features. A new feature is added as the result of multiplying both]
        [lr: The learning rate for the model]
        [n_estimators: the number of estimators for the gradient boosting regressor model]
    """
    new_features = 0
    model = GradientBoostingRegressor(learning_rate = lr, n_estimators = n_estimators)
    rskf = KFold(n_splits=5, shuffle=True,random_state=3557)
    n = 0
    losses = {}
    total_loss = 0
    # Loss without adding features
    for train_index, test_index in rskf.split(X, y):
        n+=1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        total_loss += average_absolute_loss(y_pred, y_test)
    losses[new_features] = total_loss/n
    X_new = X.copy()
    for feature in features:
        new_features += 1
        total_loss = 0
        n = 0
        X_new = add_feature(X_new, feature[0], feature[1])
        for train_index, test_index in rskf.split(X, y):
            n+=1
            X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            total_loss += average_absolute_loss(y_pred, y_test)
        losses[new_features] = total_loss/n
    return losses

def greedy_feature_removal_kfold(X, y, lr = 0.05, n_estimators = 100):
    """
        Removes features greedily and returns the average loss using k-fold cross validation for each removal.
    """
    average_losses = {}
    features = list(X.columns[:])
    removed_features = []
    rskf = KFold(n_splits=5, shuffle=True,random_state=355)
    for i in np.arange(0, X.shape[1] - 1):
        total_loss = 0
        gbr = GradientBoostingRegressor(learning_rate = lr, n_estimators = n_estimators)
        gbr.fit(X.loc[:,features], y)
        importances = gbr.feature_importances_
        imp_series = pd.Series(data=importances, index=X.loc[:,features].columns)
        imp_series.sort_values(ascending=True, inplace=True)
        removed_features.append(imp_series.index[0])
        features.remove(imp_series.index[0])
        n = 0
        for train_index, test_index in rskf.split(X, y):
            n+=1
            X_train, X_test = X.loc[:,features].iloc[train_index], X.loc[:,features].iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            total_loss += average_absolute_loss(y_pred, y_test)
        average_losses[i] = total_loss/n
            
    return removed_features, average_losses