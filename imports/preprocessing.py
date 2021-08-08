import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def bins(DC):
    if DC < 150:
        return 1
    elif DC < 500:
        return 2
    else:
        return 3

class DataPreprocessor:
    def __init__(self):
        self.transformable_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind']
    def fit(self, data):
        self.data = data.copy()
        self.remove_outlier()
        self.log_transform_RH()
        self.processor = MinMaxScaler().fit(self.data.loc[:,self.transformable_cols])
    def transform_with_2target(self):
        month_df = self.month_encoding()
        xy_df = self.xy_encoding()
        day_df = self.day_encoding()
        rain_series = self.rain_transform()
        dc_df = self.DC_encoding()
        target_df = self.area_split_transform()
        numerical_df = pd.DataFrame(self.processor.transform(self.data[self.transformable_cols]), columns = self.transformable_cols)
        output = pd.concat([month_df, xy_df, day_df, rain_series, dc_df, numerical_df, target_df], axis=1)
        return output
    def month_encoding(self):
        month_encoder = OneHotEncoder(handle_unknown='ignore')
        month_encoder.fit(self.data[["month"]])
        months = ['jan', 'feb', 'mar', 'apr','may','jun', 'jul', 'aug','sep', 'oct', 'nov', 'dec']
        months_sorted = sorted(months)
        months_array = month_encoder.transform(self.data[["month"]]).toarray()
        months_bool = [month+"_bool"for month in months_sorted]
        month_df = pd.DataFrame(months_array, columns = months_bool)
        return month_df
    def xy_encoding(self):
        df_xy = self.data.copy()
        df_xy = df_xy.assign(XY = df_xy.loc[:,"X"].astype(str) + df_xy.loc[:,"Y"].astype(str))
        xy_encoded = pd.get_dummies(df_xy["XY"])
        return xy_encoded
    def day_encoding(self):
        df_days = self.data.copy()
        days_encoded = pd.get_dummies(df_days["day"])
        return days_encoded
    def DC_encoding(self):
        series_dc = pd.Series([bins(x) for x in self.data["DC"].to_numpy()])
        series_dc.name = "DC_range"
        return series_dc
    def rain_transform(self):
        rain_bool = pd.Series((self.data["rain"] > 0).astype(int))
        return rain_bool
    def area_transform_to_log(self):
        area = self.data.loc[:,"area"]
        area = np.log(1 + area)
        return area
    def area_split_transform(self):
        area_bool = (self.data.loc[:,"area"] > 0).astype(int)
        area = self.data.loc[:,"area"]
        area = np.log(1 + area)
        area_df = pd.concat([area_bool, area], axis=1)
        area_df.columns = ["area_bool", "area"]
        return area_df
    def remove_outlier(self):
        isi = self.data.loc[:,["ISI"]]
        self.data = self.data.iloc[(isi<40).to_numpy()]
        self.data = self.data.reset_index()
    def log_transform_RH(self):
        self.data.assign(RH = np.log(self.data["RH"]))
    def znormalize_numerics(self):
        processed_cols = [self.processors[proc].transform(self.data[[proc]]) for proc in self.processors.keys()]
        [pd.Series(processed_cols[i]) for i in processed_cols]
    def save_instance(self, file):
        with open(file, "wb") as f:
            pickle.dump(self, f)
            

class DataPreprocessorPCA(DataPreprocessor):
    def __init__(self):
        pass
    def transform_with_2target(self):
        month_df = self.month_encoding()
        xy_df = self.xy_encoding()
        day_df = self.day_encoding()
        rain_series = self.rain_transform()
        dc_df = self.DC_encoding()
        target_df = self.area_split_transform()
        numerical_df = pd.DataFrame(self.processor.transform(self.data[self.transformable_cols]), columns = self.transformable_cols)
        self.pca_processor = PCA()
        self.pca_processor.fit(numerical_df)
        pca_mat = self.pca_processor.transform(numerical_df)
        numerical_df = pd.DataFrame(pca_mat, columns = ['pc'+str(i) for i in np.arange(1, 8)])        
        output = pd.concat([month_df, xy_df, day_df, rain_series, dc_df, numerical_df, target_df], axis=1)
        return output
    

class DataPreprocessorPCA_encoding2(DataPreprocessor):
    def __init__(self):
        pass
    def transform_with_2target(self):
        month_df = self.month_encoding()
        xy_df = self.xy_encoding()
        day_df = self.day_encoding()
        rain_series = self.rain_transform()
        dc_df = self.DC_encoding()
        target_df = self.area_split_transform()
        numerical_df = pd.DataFrame(self.processor.transform(self.data[self.transformable_cols]), columns = self.transformable_cols)
        self.pca_processor = PCA()
        self.pca_processor.fit(numerical_df)
        pca_mat = self.pca_processor.transform(numerical_df)
        numerical_df = pd.DataFrame(pca_mat, columns = ['pc'+str(i) for i in np.arange(1, 8)])        
        output = pd.concat([month_df, xy_df, day_df, rain_series, dc_df, numerical_df, target_df], axis=1)
        return output
    def month_encoding(self):
        dict_months = {'mar':3, 'oct':10, 'aug':8, 'sep':9, 'apr':4, 'jun':6, 'jul':7, 'feb':2, 'jan':1,'dec':12, 'may':5, 'nov':11}
        months = self.data[["month"]].replace(dict_months)
        return months
    def xy_encoding(self):
        return self.data[["X", "Y"]]
    

class DataPreprocessorSplitter(DataPreprocessor):
    def __init__(self):
        self.transformable_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind']
    def transform_with_2target(self):
        month_df15 = self.month_encoding_15()
        month_df612 = self.month_encoding_612()
        
        xy_df = self.xy_encoding()
        day_df = self.day_encoding()
        rain_series = self.rain_transform()
        dc_df = self.DC_encoding()
        target_df = self.area_split_transform()
        
        numerical_df15 = pd.DataFrame(self.processor_15.transform(self.data_jan_may[self.transformable_cols]), columns = self.transformable_cols)
        numerical_df612 = pd.DataFrame(self.processor_612.transform(self.data_jun_dec[self.transformable_cols]), columns = self.transformable_cols)
        self.pca_processor15 = PCA()
        self.pca_processor15.fit(numerical_df15)
        pca_mat15 = self.pca_processor15.transform(numerical_df15)
        numerical_df15 = pd.DataFrame(pca_mat15, columns = ['pc'+str(i) for i in np.arange(1, 8)])
        self.pca_processor612 = PCA()
        self.pca_processor612.fit(numerical_df612)
        pca_mat612 = self.pca_processor612.transform(numerical_df612)
        numerical_df612 = pd.DataFrame(pca_mat612, columns = ['pc'+str(i) for i in np.arange(1, 8)])
        
        
        common_df = pd.concat([xy_df, day_df, rain_series, dc_df, target_df], axis=1)
        output1 = pd.concat([common_df.loc[self.filter15].reset_index(), month_df15.reset_index(), numerical_df15.reset_index()], axis=1)
        output2 = pd.concat([common_df.loc[self.filter612].reset_index(), month_df612.reset_index(), numerical_df612.reset_index()], axis=1)
        return output1, output2
    def split(self):
        months1 = ['jan', 'feb', 'mar', 'apr','may']
        months2 = ['jun', 'jul', 'aug','sep', 'oct', 'nov', 'dec']
        self.filter15 = (self.data[["month"]].isin(months1)).to_numpy()
        self.filter612 = (self.data[["month"]].isin(months2)).to_numpy()
        self.data_jan_may = self.data.loc[self.filter15]
        self.data_jun_dec = self.data.loc[self.filter612]
        
    def fit(self, data):
        self.data = data.copy()
        self.log_transform_RH()
        self.remove_outlier()
        self.split()
        self.processor_15 = MinMaxScaler().fit(self.data_jan_may.loc[:,self.transformable_cols])
        self.processor_612 = MinMaxScaler().fit(self.data_jun_dec.loc[:,self.transformable_cols])
        
    def month_encoding_15(self):
        month_encoder = OneHotEncoder(handle_unknown='ignore')
        month_encoder.fit(self.data_jan_may[["month"]])
        months = ['jan', 'feb', 'mar', 'apr','may']
        months_sorted = sorted(months)
        months_array = month_encoder.transform(self.data_jan_may[["month"]]).toarray()
        months_bool = [month+"_bool"for month in months_sorted]
        month_df = pd.DataFrame(months_array, columns = months_bool)
        return month_df

    def month_encoding_612(self):
        month_encoder = OneHotEncoder(handle_unknown='ignore')
        month_encoder.fit(self.data_jun_dec[["month"]])
        months = ['jun', 'jul', 'aug','sep', 'oct', 'nov', 'dec']
        months_sorted = sorted(months)
        months_array = month_encoder.transform(self.data_jun_dec[["month"]]).toarray()
        months_bool = [month+"_bool"for month in months_sorted]
        month_df = pd.DataFrame(months_array, columns = months_bool)
        return month_df