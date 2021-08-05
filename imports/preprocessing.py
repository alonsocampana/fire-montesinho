import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        pass
    def fit(self, data):
        self.data = data.copy()
        self.transformable_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind']
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
    def log_transform_RH(self):
        self.data.assign(RH = np.log(self.data["RH"]))
    def znormalize_numerics(self):
        processed_cols = [self.processors[proc].transform(self.data[[proc]]) for proc in self.processors.keys()]
        [pd.Series(processed_cols[i]) for i in processed_cols]
    def save_instance(self, file):
        with open(file, "wb") as f:
            pickle.dump(self, f)

def bins(DC):
    if DC < 150:
        return 1
    elif DC < 500:
        return 2
    else:
        return 3
