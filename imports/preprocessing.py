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
    """
        Interface for preprocessing Montesinho data. Creates different encodings and returns two versions of the target variable.
    """
    def __init__(self):
        self.transformable_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind']

    def fit(self, data):
        """
            Stores the data and initializes the preprocessors that can be later accesed for transforming other observations.
        """
        self.data = data.copy()
        self.remove_outlier()
        self.log_transform_RH()
        self.processor = MinMaxScaler().fit(self.data.loc[:,self.transformable_cols])

    def transform_with_2target(self):
        """
            calls the encodings and creates a DataFrame with the new columns
        """
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
        """
            Stores every month as one-hot-encoded labels
        """
        month_encoder = OneHotEncoder(handle_unknown='ignore')
        month_encoder.fit(self.data[["month"]])
        months_temp = ['jan', 'feb', 'mar', 'apr','may','jun', 'jul', 'aug','sep', 'oct', 'nov', 'dec']
        months = [month for month in months_temp if month in (self.data["month"].to_numpy())]
        months_sorted = sorted(months)
        months_array = month_encoder.transform(self.data[["month"]]).toarray()
        months_bool = [month+"_bool" for month in months_sorted]
        month_df = pd.DataFrame(months_array, columns = months_bool)
        return month_df

    def xy_encoding(self):
        """
            Stores every XY parcel as one-hot-encoded labels
        """
        df_xy = self.data.copy()
        df_xy = df_xy.assign(XY = df_xy.loc[:,"X"].astype(str) + df_xy.loc[:,"Y"].astype(str))
        xy_encoded = pd.get_dummies(df_xy["XY"])
        return xy_encoded

    def day_encoding(self):
        """
            Stores every day of the week as one-hot-encoded labels
        """
        df_days = self.data.copy()
        days_encoded = pd.get_dummies(df_days["day"])
        return days_encoded

    def DC_encoding(self):
        """
            creates a one-hot encoding for three ranges of DC
        """
        series_dc = pd.Series([bins(x) for x in self.data["DC"].to_numpy()])
        series_dc.name = "DC_range"
        return series_dc

    def rain_transform(self):
        """
            Transforms rain into a boolean variable
        """
        rain_bool = pd.Series((self.data["rain"] > 0).astype(int))
        return rain_bool

    def area_transform_to_log(self):
        """
            Transforms the target varaible into logarithmic scale
        """
        area = self.data.loc[:,"area"]
        area = np.log(1 + area)
        return area

    def area_split_transform(self):
        """
            Creates a boolean where 0 means the burnt area was 0, 1 otherwise
        """
        area_bool = (self.data.loc[:,"area"] > 0).astype(int)
        area = self.data.loc[:,"area"]
        area = np.log(1 + area)
        area_df = pd.concat([area_bool, area], axis=1)
        area_df.columns = ["area_bool", "area"]
        return area_df

    def remove_outlier(self):
        isi = self.data.loc[:,["ISI"]]
        """
            removes the outliers.
        """
        self.data = self.data.iloc[(isi<40).to_numpy()]
        self.data = self.data.reset_index()

    def log_transform_RH(self):
        """
            Transforms RH into logarithmic scale
        """
        self.data.assign(RH = np.log(self.data["RH"]))

    def save_instance(self, file):
        """
            Saves the object as a pickle object that can be later accessed
        """
        with open(file, "wb") as f:
            pickle.dump(self, f)


class DataPreprocessorPCA(DataPreprocessor):
    """
        Interface for preprocessing Montesinho data. Includes additional PCA transformation of the numeric continuous variables
    """
    def __init__(self):
        self.transformable_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind']

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
    def transform_with_1target(self):
        month_df = self.month_encoding()
        xy_df = self.xy_encoding()
        day_df = self.day_encoding()
        rain_series = self.rain_transform()
        dc_df = self.DC_encoding()
        target_df = self.area_split_transform()
        target_df = target_df[["area"]]
        numerical_df = pd.DataFrame(self.processor.transform(self.data[self.transformable_cols]), columns = self.transformable_cols)
        self.pca_processor = PCA()
        self.pca_processor.fit(numerical_df)
        pca_mat = self.pca_processor.transform(numerical_df)
        numerical_df = pd.DataFrame(pca_mat, columns = ['pc'+str(i) for i in np.arange(1, 8)])
        output = pd.concat([month_df, xy_df, day_df, rain_series, dc_df, numerical_df, target_df], axis=1)
        return output
    def transform_single_instance(self, instance):
        numeric_series = pd.Series(self.processor.transform(instance[self.transformable_cols].to_numpy().reshape(1, -1))[0], index = self.transformable_cols)
        month_bools = ['apr_bool', 'aug_bool', 'dec_bool', 'feb_bool', 'jan_bool', 'jul_bool',
       'jun_bool', 'mar_bool', 'may_bool', 'nov_bool', 'oct_bool', 'sep_bool']
        XY_bools = ['12', '13', '14', '15', '22', '23', '24', '25', '33', '34', '35', '36',
       '43', '44', '45', '46', '54', '55', '56', '63', '64', '65', '66', '73',
       '74', '75', '76', '83', '84', '85', '86', '88', '94', '95', '96', '99']
        month_key = instance["month"] + "_bool"
        week_bools = ['fri', 'mon', 'sat', 'sun', 'thu', 'tue', 'wed']
        month_key = instance["month"] + "_bool"
        XY_series = pd.Series(np.zeros(len(XY_bools)), index = XY_bools)
        month_series = pd.Series(np.zeros(len(month_bools)), index = month_bools)
        month_series[month_key] = 1
        XY_key = str(instance["X"]) + str(instance["Y"])
        XY_series[XY_key] = 1
        week_series = pd.Series(np.zeros(7), index = week_bools)
        week_key = instance["day"]
        week_series[week_key] = 1
        rain_bool = (instance["rain"] != 0)
        DC_range = bins(instance["DC"])
        DC_rain_series = pd.Series({"DC_range" : bins(instance["DC"]), "rain":(int(instance["rain"] != 0))})
        encoded_series = pd.concat([XY_series, month_series, week_series, DC_rain_series])
        pca_mat = self.pca_processor.transform(numeric_series.to_numpy().reshape(1, -1))
        numerical_series = pd.Series(pca_mat[0], index = ['pc'+str(i) for i in np.arange(1, 8)])
        output = XY_series.append([month_series, week_series, DC_rain_series, numerical_series])
        return  output


class DataPreprocessorPCA_encoding2(DataPreprocessor):
    """
        Interface for preprocessing Montesinho data. Includes additional PCA transformation of the numeric continuous variables and a different encoding of the variables
    """
    def __init__(self):
        self.transformable_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind']

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
    """
        Interface for preprocessing Montesinho data. Includes additional PCA transformation of the numeric continuous variables and splits the dataset in function of the months
    """
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
        #create a dataframe for each subset to be normalized independently
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
        """
            Splits the data in function of the month of the observation
        """
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


class DataPreprocessorSplitter2(DataPreprocessorSplitter):
    """
        Interface for preprocessing Montesinho data. Includes additional PCA transformation of the numeric continuous variables and splits the dataset in function of the months, and the resulting dataset for june-december in function of the X coordinate
    """
    def __init__(self):
        self.transformable_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind']
    def transform_with_2target(self):
        """
            Transforms the data: Creates the corresponding new encodings, normalizes the continuous variables, PCA rotates it, and concatenates it together in 3 DataFrames.
        """
        month_df15 = self.month_encoding_15()
        month_df612 = self.month_encoding_612()
        #calls the corresponding encoders
        xy_df = self.xy_encoding()
        day_df = self.day_encoding()
        rain_series = self.rain_transform()
        dc_df = self.DC_encoding()
        target_df = self.area_split_transform()
        # creates the numerical dataframes for each subset of the output
        numerical_df15 = pd.DataFrame(self.processor_15.transform(self.data_jan_may[self.transformable_cols]), columns = self.transformable_cols)
        numerical_df612_15 = pd.DataFrame(self.processor_612_15.transform(self.data_jun_dec_15[self.transformable_cols]), columns = self.transformable_cols)
        numerical_df612_69 = pd.DataFrame(self.processor_612_69.transform(self.data_jun_dec_69[self.transformable_cols]), columns = self.transformable_cols)
        # creates processors that can be later used for transforming observations
        self.pca_processor15 = PCA()
        self.pca_processor15.fit(numerical_df15)
        pca_mat15 = self.pca_processor15.transform(numerical_df15)
        numerical_df15 = pd.DataFrame(pca_mat15, columns = ['pc'+str(i) for i in np.arange(1, 8)])
        # creates processors that can be later used for transforming observations
        self.pca_processor612_15 = PCA()
        self.pca_processor612_15.fit(numerical_df612_15)
        pca_mat612_15 = self.pca_processor612_15.transform(numerical_df612_15)
        numerical_df612_15 = pd.DataFrame(pca_mat612_15, columns = ['pc'+str(i) for i in np.arange(1, 8)])
        # creates processors that can be later used for transforming observations
        self.pca_processor612_69 = PCA()
        self.pca_processor612_69.fit(numerical_df612_69)
        pca_mat612_69 = self.pca_processor612_69.transform(numerical_df612_69)
        numerical_df612_69 = pd.DataFrame(pca_mat612_69, columns = ['pc'+str(i) for i in np.arange(1, 8)])
        # Creates the corresponding three dataframes that are going to be the output
        common_df = pd.concat([xy_df, day_df, rain_series, dc_df, target_df], axis=1)
        output1 = pd.concat([common_df.loc[self.filter15].reset_index(), month_df15.reset_index(), numerical_df15.reset_index()], axis=1)
        output2 = pd.concat([common_df.loc[self.filterx15 & self.filter612].reset_index(), month_df612.loc[self.filterx15_split].reset_index(), numerical_df612_15.reset_index()], axis=1)
        output3 = pd.concat([common_df.loc[self.filterx69 & self.filter612].reset_index(), month_df612.loc[self.filterx69_split].reset_index(), numerical_df612_69.reset_index()], axis=1)
        return output1, output2, output3
    def split(self):
        months1 = ['jan', 'feb', 'mar', 'apr','may']
        months2 = ['jun', 'jul', 'aug','sep', 'oct', 'nov', 'dec']
        #creates the boolean filters to subset logically the different datasets
        self.filter15 = (self.data[["month"]].isin(months1)).to_numpy()
        self.filter612 = (self.data[["month"]].isin(months2)).to_numpy()
        self.filterx15 = (self.data[["X"]] < 6).to_numpy()
        self.filterx69 = (self.data[["X"]] >= 6).to_numpy()
        self.data_jan_may = self.data.loc[self.filter15]
        self.data_jun_dec = self.data.loc[self.filter612]
        self.filterx15_split = (self.data_jun_dec[["X"]] < 6).to_numpy()
        self.filterx69_split = (self.data_jun_dec[["X"]] >= 6).to_numpy()
        self.data_jun_dec_15 = self.data.loc[self.filter612 & self.filterx15]
        self.data_jun_dec_69 = self.data.loc[self.filter612 & self.filterx69]

    def fit(self, data):
        self.data = data.copy()
        self.log_transform_RH()
        self.remove_outlier()
        self.split()
        self.processor_15 = MinMaxScaler().fit(self.data_jan_may.loc[:,self.transformable_cols])
        self.processor_612_15 = MinMaxScaler().fit(self.data_jun_dec_15.loc[:,self.transformable_cols])
        self.processor_612_69 = MinMaxScaler().fit(self.data_jun_dec_69.loc[:,self.transformable_cols])
