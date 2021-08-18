from exploratory_analysis import *
from preprocessing import *
from model_selection import *
import pickle

class MontesinhoCompleteModel_evaluator():
    """
        Highly complex model doing previous classification
    """
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
    
    def get_model(self):
        return self.model
    
    def average_loss(self):
        loss1 = self.eval_data1()
        loss2 = self.eval_data2()
        loss3 = self.eval_data3()
        total = self.l1 + self.l2 + self.l3
        return (self.l1 * loss1 + self.l2 * loss2 + self.l3 * loss3)/total
    
class MontesinhoCompleteModel_evaluator_simple():
    """
        End to end simplified model
    """
    def __init__(self, model, degree):
        """
        [model: A valid sklearn model]
        [degree: the degree of the polynomial featurizer for the model]
        """
        self.preprocessor = DataPreprocessorPCA()
        self.poly = PolynomialFeatures(degree=degree)
        self.model = model

    def fit(self, data):
        self.preprocessor.fit(data)
        self.data = self.preprocessor.transform_with_1target()
        self.poly.fit(self.data.drop(["area"], axis=1))
        self.data_poly = self.poly.transform(self.data.drop(["area"], axis=1))
        
    def train(self):
        """
            trains the model on the whole data
        """
        X = self.data_poly
        self.columns = self.data.drop(["area"], axis=1).columns
        y= self.data["area"]
        self.model.fit(X, y)
        
    def predict_instance(self, instance): 
        """
            Predicts the value for a single instance, a dataframe containing the same columns as the original csv file
        """
        transformed_instance = self.preprocessor.transform_single_instance(instance)[self.columns]
        poly_instance = self.poly.transform(transformed_instance.to_numpy().reshape(1, -1))
        prediction = self.model.predict(poly_instance)
        return np.exp(prediction[0]) - 1
    
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
    
    def get_model(self):
        return self.model
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        

class MontesinhoCompleteModel_evaluator_fe(MontesinhoCompleteModel_evaluator_simple):
    """
        End to end model with feature engineering.
    """
    def __init__(self, model, degree, feature_added, feature_removed):
        """
        [model: A valid sklearn model]
        [degree: the degree of the polynomial featurizer for the model]
        [feature_added: An array containing pairs of variables to be added to the model as new features]
        [feature_removed: A list containing features to be removed from the model]
        """
        self.feature_added = feature_added
        self.feature_removed = feature_removed
        self.preprocessor = DataPreprocessorPCA()
        self.poly = PolynomialFeatures(degree=degree)
        self.model = model
        
    def train(self):
        """
            trains the model on the whole data
        """
        X = self.data_poly
        self.columns = self.data.drop(["area"], axis=1).columns
        y= self.data["area"]
        self.model.fit(X, y)

    def fit(self, data):
        """
            Performs the initial fit of the preprocessors.
        """
        self.preprocessor.fit(data)
        self.data = self.preprocessor.transform_with_1target()
        for feature in self.feature_added:
            self.data = add_feature(self.data, feature[0], feature[1])
        self.feature_removed = [feat for feat in self.feature_removed if feat in self.data.columns]
        self.data = self.data.drop(self.feature_removed, axis=1)
        self.poly.fit(self.data.drop(["area"], axis=1))
        self.data_poly = self.poly.transform(self.data.drop(["area"], axis=1))
        
    def predict_instance(self, instance):
        """
            Predicts the value for a single instance, a dataframe containing the same columns as the original csv file
        """
        transformed_instance = self.preprocessor.transform_single_instance(instance)
        for feature in self.feature_added:
            transformed_instance = add_feature_series(transformed_instance, feature[0], feature[1])
        transformed_instance = transformed_instance.drop(self.feature_removed)[self.columns]
        poly_instance = self.poly.transform(transformed_instance.to_numpy().reshape(1, -1))
        prediction = self.model.predict(poly_instance)
        return np.exp(prediction[0]) - 1

class MontesinhoCompleteModel_evaluator_sk(MontesinhoCompleteModel_evaluator_fe):
    """
        End to end model with bayesian optimization.
    """
    def __init__(self, model, degree, feature_added, feature_removed, space):
        """
        [model: A valid sklearn model]
        [degree: the degree of the polynomial featurizer for the model]
        [feature_added: An array containing pairs of variables to be added to the model as new features]
        [feature_removed: A list containing features to be removed from the model]
        """
        self.feature_added = feature_added
        self.feature_removed = feature_removed
        self.preprocessor = DataPreprocessorPCA()
        self.poly = PolynomialFeatures(degree=degree)
        self.model = model
        self.space = space

    def fit(self, data):
        """
            Performs the initial fit of the preprocessors.
        """
        self.preprocessor.fit(data)
        self.data = self.preprocessor.transform_with_1target()
        for feature in self.feature_added:
            self.data = add_feature(self.data, feature[0], feature[1])
        self.data = self.data.drop(self.feature_removed, axis=1)
        self.poly.fit(self.data.drop(["area"], axis=1))
        self.data_poly = self.poly.transform(self.data.drop(["area"], axis=1))
        self.score, self.model = nested_crossval_model(self.data_poly, self.data["area"],self.model,self.space)
        
    def evaluate(self):
        return self.score
    

def add_feature_series(series, col1, col2):
    """
        creates a new feature for the series resulting from the multiplication of col1 and col2
    """
    new_name = col1 + " * " + col2
    val = series[col1] * series[col2]
    dict_out = {}
    dict_out[new_name] = val
    return (series.append(pd.Series(dict_out)))

def create_dfs(model,data, indexes_0, indexes_0_10, indexes_10_40, indexes_40_plus):
    df_0 = pd.DataFrame(columns =["obs", "pred"])
    df_0_10 = pd.DataFrame(columns =["obs", "pred"])
    df_10_40 = pd.DataFrame(columns =["obs", "pred"])
    df_40_plus = pd.DataFrame(columns =["obs", "pred"])
    for i in indexes_0:
        df_0 = df_0.append({"obs":data["area"].iloc[i], "pred":model.predict_instance(data.iloc[i])}, ignore_index=True)
    for i in indexes_0_10:
        df_0_10 = df_0_10.append({"obs":data["area"].iloc[i], "pred":model.predict_instance(data.iloc[i])}, ignore_index=True)
    for i in indexes_10_40:
        df_10_40 = df_10_40.append({"obs":data["area"].iloc[i], "pred":model.predict_instance(data.iloc[i])}, ignore_index=True)
    for i in indexes_40_plus:
        df_40_plus = df_40_plus.append({"obs":data["area"].iloc[i], "pred":model.predict_instance(data.iloc[i])}, ignore_index=True)
    return df_0,df_0_10, df_10_40,df_40_plus