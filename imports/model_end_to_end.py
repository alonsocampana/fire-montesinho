from exploratory_analysis import *
from preprocessing import *
from model_selection import *
import pickle

class MontesinhoCompleteModel_evaluator():
    def __init__(self, classifier_jan, classifier_jun_15, classifier_jun_69,
                 featurizer_jan, featurizer_jun15, featurizer_jun69, regressor_jan,
                 regressor_jun_15, regressor_jun_69):
        self.preprocessor = DataPreprocessorSplitter2()
        self.classifier_jan = classifier_jan
        self.classifier_jun_15 = classifier_jun_15
        self.classifier_jun_69 = classifier_jun_69
        self.featurizer_jan = featurizer_jan
        self.featurizer_jun15 = featurizer_jun15
        self.featurizer_jun69 = featurizer_jun69
        self.regressor_jan = regressor_jan
        self.regressor_jun_15 = regressor_jun_15
        self.regressor_jun_69 = regressor_jun_69

    def fit(self, data):
        self.preprocessor.fit(data)
        self.data1, self.data2, self.data3 = self.preprocessor.transform_with_2target()
        self.data1, self.data2, self.data3 = self.data1.drop(['index'], axis=1), self.data2.drop(['index'], axis=1), self.data3.drop(['index'], axis=1)
        self.l1, self.l2, self.l3 =  self.data1.shape[0], self.data2.shape[0], self.data3.shape[0]
        
    def eval_data1(self):
        rskf = KFold(n_splits=5, shuffle=True,random_state=3558)
        X = self.data1.drop(["area", "area_bool"], axis=1)
        self.featurizer_jan.fit(X)
        Xt = self.featurizer_jan.transform(X)
        y1 = self.data1["area"]
        y2 = self.data1["area_bool"]
        losses = []
        n = 0
        for train_index, test_index in rskf.split(X, y1):
            temp_losses = 0
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Xt_train, Xt_test = Xt[train_index], Xt[test_index]
            y1_train, y1_test = y1.iloc[train_index], y1.iloc[test_index]
            y2_train, y2_test = y2.iloc[train_index], y2.iloc[test_index]
            self.classifier_jan.fit(X_train, y2_train)
            filter_nonzero = (y2_train == 1)
            self.regressor_jan.fit(Xt_train[filter_nonzero], y1_train[filter_nonzero])
            y2_pred = self.classifier_jan.predict(X_test)
            false_negatives = ((y2_pred == 0) & (y2_test ==1))
            temp_losses += np.sum(y1_test[false_negatives])
            positives = (y2_pred == 1)
            if sum(positives) > 0:
                y1_pred = self.regressor_jan.predict(Xt_test[positives])
                temp_losses += np.sum(abs(y1_pred - y1_test[positives]))
            n += 1
            losses.append(temp_losses/X_test.shape[0])
        return (sum(losses)/n)
    
    def eval_data2(self):
        rskf = KFold(n_splits=5, shuffle=True,random_state=3558)
        X = self.data2.drop(["area", "area_bool"], axis=1)
        self.featurizer_jun15.fit(X)
        Xt = self.featurizer_jun15.transform(X)
        y1 = self.data2["area"]
        y2 = self.data2["area_bool"]
        losses = []
        n = 0
        for train_index, test_index in rskf.split(X, y1):
            temp_losses = 0
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Xt_train, Xt_test = Xt[train_index], Xt[test_index]
            y1_train, y1_test = y1.iloc[train_index], y1.iloc[test_index]
            y2_train, y2_test = y2.iloc[train_index], y2.iloc[test_index]
            self.classifier_jun_15.fit(X_train, y2_train)
            filter_nonzero = (y2_train == 1)
            self.regressor_jun_15.fit(Xt_train[filter_nonzero], y1_train[filter_nonzero])
            y2_pred = self.classifier_jun_15.predict(X_test)
            false_negatives = ((y2_pred == 0) & (y2_test ==1))
            temp_losses += np.sum(y1_test[false_negatives])
            positives = (y2_pred == 1)
            if sum(positives) > 0:
                y1_pred = self.regressor_jun_15.predict(Xt_test[positives])
                temp_losses += np.sum(abs(y1_pred - y1_test[positives]))
            n += 1
            losses.append(temp_losses/X_test.shape[0])
        return (sum(losses)/n)
    
    def eval_data3(self):
        rskf = KFold(n_splits=5, shuffle=True,random_state=3558)
        X = self.data3.drop(["area", "area_bool"], axis=1)
        self.featurizer_jun69.fit(X)
        Xt = self.featurizer_jun69.transform(X)
        y1 = self.data3["area"]
        y2 = self.data3["area_bool"]
        losses = []
        n = 0
        for train_index, test_index in rskf.split(X, y1):
            temp_losses = 0
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Xt_train, Xt_test = Xt[train_index], Xt[test_index]
            y1_train, y1_test = y1.iloc[train_index], y1.iloc[test_index]
            y2_train, y2_test = y2.iloc[train_index], y2.iloc[test_index]
            self.classifier_jun_69.fit(X_train, y2_train)
            filter_nonzero = (y2_train == 1)
            self.regressor_jun_69.fit(Xt_train[filter_nonzero], y1_train[filter_nonzero])
            y2_pred = self.classifier_jun_69.predict(X_test)
            false_negatives = ((y2_pred == 0) & (y2_test ==1))
            temp_losses += np.sum(y1_test[false_negatives])
            positives = (y2_pred == 1)
            if sum(positives) > 0:
                y1_pred = self.regressor_jun_69.predict(Xt_test[positives])
                temp_losses += np.sum(abs(y1_pred - y1_test[positives]))
            n += 1
            losses.append(temp_losses/X_test.shape[0])
        return (sum(losses)/n)
    
    def average_loss(self):
        loss1 = self.eval_data1()
        loss2 = self.eval_data2()
        loss3 = self.eval_data3()
        total = self.l1 + self.l2 + self.l3
        return (self.l1 * loss1 + self.l2 * loss2 + self.l3 * loss3)/total
    
class MontesinhoCompleteModel_evaluator_simple_gbr():
    def __init__(self, model, degree):
        self.preprocessor = DataPreprocessorPCA()
        self.poly = PolynomialFeatures(degree=degree)
        self.model = model

    def fit(self, data):
        self.preprocessor.fit(data)
        self.data = self.preprocessor.transform_with_1target()
        self.poly.fit(self.data.drop(["area"], axis=1))
        self.data_poly = self.poly.transform(self.data.drop(["area"], axis=1))
    def evaluate(self):
        X = self.data_poly
        y= self.data["area"]
        rskf = KFold(n_splits=15, shuffle=True,random_state=3558)
        losses = []
        n=0
        for train_index, test_index in rskf.split(X, y):
            n += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            losses.append(average_absolute_loss(y_pred, y_test))
        return sum(losses)/n