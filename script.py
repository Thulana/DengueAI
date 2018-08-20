# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 21:25:32 2018

@author: Praveen
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import Imputer


pd.set_option('display.max_columns', 100)

mpl.rc(group='figure', figsize=(10,8))
plt.style.use('seaborn')


X_train = pd.read_csv('F:/Uni/Semester 7/Data Mining/DengAI/dengue_features_train.csv')
X_train.week_start_date = pd.to_datetime(X_train.week_start_date)
print(f'X_train: {X_train.shape}')

y_train = pd.read_csv('F:/Uni/Semester 7/Data Mining/DengAI/dengue_labels_train.csv', 
                      usecols=['total_cases'])
print(f'y_train: {y_train.shape}')

X_test = pd.read_csv('F:/Uni/Semester 7/Data Mining/DengAI/dengue_features_test.csv')
X_test.week_start_date = pd.to_datetime(X_test.week_start_date)
print(f'X_test: {X_test.shape}')

Xy_train = pd.concat([y_train, X_train], axis=1) 
print(f'Xy_train: {Xy_train.shape}')

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

X_train_sj = X_train.loc[X_train.city == 'sj', :].copy()
X_train_iq = X_train.loc[X_train.city == 'iq', :].copy()

y_train_sj = y_train.loc[X_train.city == 'sj', :].copy()
y_train_iq = y_train.loc[X_train.city == 'iq', :].copy()

X_test_sj = X_test.loc[X_test.city == 'sj', :].copy()
X_test_iq = X_test.loc[X_test.city == 'iq', :].copy()

keys = ['city', 'year', 'weekofyear']

time_series_index = ['week_start_date']

all_features = ['ndvi_ne', 'ndvi_nw',
       'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm']

selected_features_sj = [
'lag_reanalysis_specific_humidity_g_per_kg',
 'mean_station_max_temp_c',
 'weekofyear',
 'reanalysis_dew_point_temp_k'
 #        "sum_station_precip_mm",
#"mean_station_max_temp_c",
#"sum_reanalysis_sat_precip_amt_mm",
#"sum_precipitation_amt_mm",
#"mean_reanalysis_specific_humidity_g_per_kg",
#"mean_reanalysis_dew_point_temp_k",
#"mean_station_min_temp_c",
#"sum_reanalysis_precip_amt_kg_per_m2",
#"mean_station_avg_temp_c",
#"mean_reanalysis_relative_humidity_percent",
#"weekofyear",
#"mean_reanalysis_tdtr_k",
#"mean_reanalysis_min_air_temp_k",
#"mean_reanalysis_max_air_temp_k",
#"mean_reanalysis_avg_temp_k",
#"mean_reanalysis_air_temp_k",
#"ndvi_nw",
#"reanalysis_dew_point_temp_k",
#"mean_ndvi_se",
#"lag_reanalysis_dew_point_temp_k"

# score 21        
#"weekofyear",'mean_reanalysis_dew_point_temp_k', 'mean_reanalysis_specific_humidity_g_per_kg', 'mean_station_max_temp_c', 'sum_reanalysis_sat_precip_amt_mm', 'sum_precipitation_amt_mm', 'sum_station_precip_mm', 'mean_station_min_temp_c', 'mean_station_avg_temp_c', 'mean_reanalysis_relative_humidity_percent', 'mean_reanalysis_tdtr_k'
                     
]

selected_features_iq = [
'lag_reanalysis_specific_humidity_g_per_kg',
 'mean_reanalysis_specific_humidity_g_per_kg',
 'reanalysis_dew_point_temp_k',
 'reanalysis_specific_humidity_g_per_kg'
 
#        "mean_ndvi_se",
#"mean_reanalysis_min_air_temp_k",
#"mean_station_avg_temp_c",
#"sum_reanalysis_precip_amt_kg_per_m2",
#"mean_reanalysis_avg_temp_k",
#"weekofyear",
#"mean_station_max_temp_c",
#"mean_station_diur_temp_rng_c",
#"sum_reanalysis_sat_precip_amt_mm",
#"mean_ndvi_ne",
#"lag_station_max_temp_c",
#"mean_station_min_temp_c",
#"station_max_temp_c",
#"mean_reanalysis_tdtr_k",
#"reanalysis_specific_humidity_g_per_kg",
#"sum_precipitation_amt_mm",
#"sum_station_precip_mm",
#"mean_reanalysis_specific_humidity_g_per_kg",
#"mean_reanalysis_dew_point_temp_k",
#"mean_ndvi_nw"
                     
# score 21                  
#"weekofyear",'mean_ndvi_se', 'mean_reanalysis_min_air_temp_k', 'mean_station_diur_temp_rng_c', 'mean_station_max_temp_c', 'mean_station_avg_temp_c', 'sum_station_precip_mm', 'reanalysis_specific_humidity_g_per_kg', 'lag_station_min_temp_c', 'station_min_temp_c'
                     ]



lag_features = ["lag_"+ x for x in all_features ]

time_series_features = []
for feature in all_features:
    if "precip" in feature:
        time_series_features.append("sum_"+feature)
    else:
        time_series_features.append("mean_"+feature)

features_used_sj = list(set(selected_features_sj).union(set(time_series_index)).union(set(keys)))
print('features_used for San Juan:\n', sorted(features_used_sj))
features_used_iq = list(set(selected_features_iq).union(set(time_series_index)).union(set(keys)))
print('features_used Iquitos:\n', sorted(features_used_iq))

impute_columns = ['reanalysis_avg_temp_c', 'reanalysis_max_air_temp_c', 
                  'reanalysis_min_air_temp_c']

def impute_redundant_features(df):
    # Convert temperature from kelvin to celcius so we can attempt to correct 
    # for the difference between the two redundant temperature features 
    # (i.e. reanalysis and station).
    df['reanalysis_avg_temp_c'] = df.reanalysis_avg_temp_k - 273.15
    df.reanalysis_avg_temp_c -= (df.reanalysis_avg_temp_c - df.station_avg_temp_c).mean()
    df.loc[df.station_avg_temp_c.isnull(), 'station_avg_temp_c'] = df.reanalysis_avg_temp_c

    df['reanalysis_max_air_temp_c'] = df.reanalysis_max_air_temp_k - 273.15
    df.reanalysis_max_air_temp_c -= (df.reanalysis_max_air_temp_c - df.station_max_temp_c).mean()
    df.loc[df.station_max_temp_c.isnull(), 'station_max_temp_c'] = df.reanalysis_max_air_temp_c

    df['reanalysis_min_air_temp_c'] = df.reanalysis_min_air_temp_k - 273.15
    df.reanalysis_min_air_temp_c -= (df.reanalysis_min_air_temp_c - df.station_min_temp_c).mean()
    df.loc[df.station_min_temp_c.isnull(), 'station_min_temp_c'] = df.reanalysis_min_air_temp_c
    
    # Drop the temporary columns that we just added
    df.drop(impute_columns, axis=1, inplace=True)
    
    return df

X_train_sj = impute_redundant_features(X_train_sj)
X_train_iq = impute_redundant_features(X_train_iq)

X_test_sj = impute_redundant_features(X_test_sj)
X_test_iq = impute_redundant_features(X_test_iq)

def lagFeatures(df, lag):
    for feature in lag_features:
        df[feature] = df[feature[4:]].shift(lag)
        for i in range(lag):
            df[feature][i] = df[feature[4:]][0] 
    return df

def impute_missing_values(df, imputer, sum_window_size):
#    imputer.fit(df[all_features+lag_features])
#    df[all_features+lag_features] = imputer.transform(df[all_features+lag_features])
    for feature in all_features:
        if "mean_"+feature in df.columns:
            df.loc[df[feature].isnull(), feature] = df["mean_"+feature]
        else:
            df.loc[df[feature].isnull(), feature] = df["sum_"+feature]/sum_window_size
    return df

def smoothen_mean(df, window):
    roll_df = df.rolling(window=window, min_periods=1, center=True)
    for feature in time_series_features:
        if "sum_" not in feature:
            df[feature] = roll_df[feature[5:]].mean()
    return df

def smoothen_sum(df, window):
    roll_df = df.rolling(window=window, min_periods=1, center=True)
    for feature in time_series_features:
        if "sum_" in feature:
            df[feature] = roll_df[feature[4:]].sum()
    return df

def impute_time_series(df):
    for feature in time_series_features:
        for i in range(len(df.index)):
            if math.isnan(df[feature][i]):
                prev = df[feature][i-1]
                distances = df[feature]
                distances = (distances-prev).abs().sort_values().reset_index()
                for j in range(100):
                    similar_value_index = distances["index"][j+1]
                    if not math.isnan(df[feature][similar_value_index+1]):
                        df[feature][i] = df[feature][similar_value_index+1]
                        break
    return df

#def smoothen_sum(df, window):
#    roll_df = df.rolling(window=window, min_periods=1)
#    for feature in time_series_features:
#        if "sum" in feature:
#            df[feature] = roll_df[feature[4:]].mean()
#        else:
#            df[feature] = roll_df[feature[5:]].sum()
#    return df

X_train_sj["set"] = "train"
X_test_sj["set"] = "test"
X_train_iq["set"] = "train"
X_test_iq["set"] = "test"

X_all_sj = pd.concat([X_train_sj, X_test_sj], axis=0, ignore_index=True)
X_all_iq = pd.concat([X_train_iq, X_test_iq], axis=0, ignore_index=True)        

X_train_sj = X_all_sj.loc[X_all_sj.set=="train",:].copy()
X_test_sj = X_all_sj.loc[X_all_sj.set=="test",:].copy()
X_train_iq = X_all_iq.loc[X_all_iq.set=="train",:].copy()
X_test_iq = X_all_iq.loc[X_all_iq.set=="test",:].copy()


X_all_sj = pd.concat([X_train_sj, X_test_sj], axis=0, ignore_index=True)
X_all_iq = pd.concat([X_train_iq, X_test_iq], axis=0, ignore_index=True) 

sj_win = {}
iq_win = {}  

#for i in range(1, 51):  
window_size = 3 
X_all_sj = smoothen_mean(X_all_sj, window=window_size)  #106,58
X_all_iq = smoothen_mean(X_all_iq, window=window_size)  #71,139 

X_all_sj = smoothen_sum(X_all_sj, window=window_size)  #19
X_all_iq = smoothen_sum(X_all_iq, window=window_size)  # 48

X_all_sj = impute_time_series(X_all_sj)
X_all_iq = impute_time_series(X_all_iq) 

imputer_sj = Imputer(strategy='mean')
imputer_iq = Imputer(strategy='mean')
X_all_sj = impute_missing_values(X_all_sj, imputer_sj, window_size)  #19
X_all_iq = impute_missing_values(X_all_iq, imputer_sj, window_size)  # 48

window_size = 3
X_all_sj = smoothen_mean(X_all_sj, window=window_size)  #106,58
X_all_iq = smoothen_mean(X_all_iq, window=window_size)  #71,139 

X_all_sj = smoothen_sum(X_all_sj, window=window_size)  #19
X_all_iq = smoothen_sum(X_all_iq, window=window_size)  # 48

X_all_sj = lagFeatures(X_all_sj, 3)
X_all_iq = lagFeatures(X_all_iq, 3)

X_train_sj = X_all_sj.loc[X_all_sj.set=="train",:].copy()
X_test_sj = X_all_sj.loc[X_all_sj.set=="test",:].copy()
X_train_iq = X_all_iq.loc[X_all_iq.set=="train",:].copy()
X_test_iq = X_all_iq.loc[X_all_iq.set=="test",:].copy()

Xy_sj = pd.concat([X_train_sj, y_train_sj], axis=1)
X_train_iq.reset_index(drop=True, inplace=True)
y_train_iq.reset_index(drop=True, inplace=True)
Xy_iq = pd.concat([X_train_iq, y_train_iq], axis=1)

X_train_sj = X_train_sj.drop("set", axis=1)
X_test_sj = X_test_sj.drop("set", axis=1)
X_train_iq = X_train_iq.drop("set", axis=1)
X_test_iq = X_test_iq.drop("set", axis=1)

def drop_unnecessary_features(df, city):
    if city == "sj":
        return df[selected_features_sj]
#        return df[all_features+lag_features+time_series_features + ["weekofyear"]]
#        return df[lag_features]
    elif city == "iq":
        return df[selected_features_iq]
#        return df[all_features+lag_features+time_series_features + ["weekofyear"]]
#        return df[lag_features]
    else:
        return None
    
def normalize(feature):
    return (feature - feature.mean()) / feature.std()

def train_predict_score(reg, X, y):
    reg.fit(X, y)
    y_pred = reg.predict(X)
    return mean_absolute_error(y_true=y, y_pred=y_pred)

def cross_validate_out_of_sample(reg, X_train, y_train, X_cross, y_cross):
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_cross)
    y_train_pred = reg.predict(X_train)
    print("train error - " + str(mean_absolute_error(y_true=y_train, y_pred=y_train_pred)))
    return mean_absolute_error(y_true=y_cross, y_pred=y_pred)

def findsubsets(S,m):
    return set(itertools.combinations(S, m))

#plt.matshow(X_train_sj.corr())

predict_sj = X_test_sj[keys].copy()
predict_iq = X_test_iq[keys].copy()

X_train_sj = drop_unnecessary_features(X_train_sj, "sj")
X_test_sj = drop_unnecessary_features(X_test_sj, "sj")
X_train_iq = drop_unnecessary_features(X_train_iq, "iq")
X_test_iq = drop_unnecessary_features(X_test_iq, "iq")


#selected = []
#
#for i in range(10):
#    rf = RandomForestRegressor(n_estimators=5, n_jobs=-1, random_state=67)
#    rf.fit(X_train_sj, y_train_sj.total_cases)
#    features =  X_train_sj.columns
#    importances = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), features), reverse=True)
##    for entry in importances:
##        print(entry)
#    selected.append(importances[0][1])
#    X_train_sj = X_train_sj.drop(importances[0][1], axis=1)
#    
#    
#print(selected)


#X_train, X_test, y_train, y_test = train_test_split(X_train_sj, y_train_sj, test_size=0.4, random_state=0)
#clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
#clf.fit(X_train, y_train)
#importanceDict = {}
#importanceList = []
#for feature in zip(all_features+lag_features+time_series_features, clf.feature_importances_):
#    importanceDict[feature[1]] = feature[0]
#    importanceList.append(feature[1])
#importanceList.sort()
#for value in importanceList:
#    print(str(importanceDict[value]) + " - " + str(value))

        
    



features_to_normalize_sj = selected_features_sj
features_to_normalize_iq = selected_features_iq

X_train_sj[features_to_normalize_sj] = X_train_sj[features_to_normalize_sj].apply(normalize, axis=0)
X_train_iq[features_to_normalize_iq] = X_train_iq[features_to_normalize_iq].apply(normalize, axis=0)
X_test_sj[features_to_normalize_sj] = X_test_sj[features_to_normalize_sj].apply(normalize, axis=0)
X_test_iq[features_to_normalize_iq] = X_test_iq[features_to_normalize_iq].apply(normalize, axis=0)

X_train_sj_cross, X_cross_sj, y_train_sj_cross, y_cross_sj = train_test_split(X_train_sj, 
                                                                  y_train_sj,
                                                                  test_size=0.2,
                                                                  stratify=X_train_sj.weekofyear)
##
X_train_iq_cross, X_cross_iq, y_train_iq_cross, y_cross_iq = train_test_split(X_train_iq, 
                                                                  y_train_iq, 
                                                                  test_size=0.2,
#                                                                  stratify=X_train_iq.weekofyear
                                                                  )

#for i in range(15,16):
#reg_sj = GradientBoostingRegressor(learning_rate=0.01, max_depth=7, n_estimators=1000, random_state=67)
reg_sj = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=67, max_depth=5, criterion='mae')
#reg_sj = LinearRegression()
print(-cross_val_score(reg_sj,X_train_sj,y_train_sj.total_cases,cv=10, scoring="neg_mean_absolute_error").mean())
print(cross_validate_out_of_sample(reg_sj, X_train_sj_cross, y_train_sj_cross.total_cases, X_cross_sj, y_cross_sj.total_cases))
#sj_win[i] = cross_validate_out_of_sample(reg_sj, X_train_sj_cross, y_train_sj_cross.total_cases, X_cross_sj, y_cross_sj.total_cases)


#reg_iq = GradientBoostingRegressor(learning_rate=0.01, max_depth=7, n_estimators=1000, random_state=67)
reg_iq = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=67, max_depth=3, criterion='mae')
#reg_iq = LinearRegression()
print(-cross_val_score(reg_iq,X_train_iq,y_train_iq.total_cases,cv=10, scoring="neg_mean_absolute_error").mean())
print(cross_validate_out_of_sample(reg_iq, X_train_iq_cross, y_train_iq_cross.total_cases, X_cross_iq, y_cross_iq.total_cases))
#iq_win[i] = cross_validate_out_of_sample(reg_iq, X_train_iq_cross, y_train_iq_cross.total_cases, X_cross_iq, y_cross_iq.total_cases)

#min_sj = min(sj_win, key=sj_win.get)
#min_iq = min(iq_win, key=iq_win.get)
#print("sj best window size = " + str(min_sj) + " - " + str(sj_win[min_sj]))
#print("iq best window size = " + str(min_iq) + " - " + str(iq_win[min_iq]))


#reg_sj = GradientBoostingRegressor(learning_rate=0.1, max_depth=5, n_estimators=500, random_state=67)
reg_sj = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=67, max_depth=5, criterion='mae')
reg_sj.fit(X_train_sj, y_train_sj.total_cases)

#reg_iq = GradientBoostingRegressor(learning_rate=0.1, max_depth=5, n_estimators=500, random_state=67)
reg_iq = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=67, max_depth=3, criterion='mae')
reg_iq.fit(X_train_iq, y_train_iq.total_cases)

y_sj_pred = reg_sj.predict(X_test_sj)
predict_sj['total_cases'] = y_sj_pred.round().astype(int)

y_iq_pred = reg_iq.predict(X_test_iq)
predict_iq['total_cases'] = y_iq_pred.round().astype(int)

predict_df = pd.concat([predict_sj, predict_iq], axis=0)

predict_df.loc[predict_df.total_cases < 0, 'total_cases'] = 0

submission_filename = 'dengue_submission_001.csv'
predict_df.to_csv(submission_filename, index=False)






