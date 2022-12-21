#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as minmax 
from sklearn.tree import DecisionTreeClassifier as dt_c
from sklearn.ensemble import RandomForestClassifier as rf_c
from sklearn.ensemble import GradientBoostingClassifier as gb_c 
# !pip install catboost : 사전에 설치 필요
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans


# 시각화 
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False
ㄴ

# 데이터 로딩 및 데이터 분리
df1 = pd.read_csv('그리드100_221107_final.csv', encoding='cp949')
df1 = df1.drop('Unnamed: 0', axis=1)


df_modeling = df1[df1['음주사고수'] > 0]
df_scoring = df1[df1['음주사고수'] == 0]


df_modeling_x = df_modeling[['가게면적', 'CCTV수', '교차로수', '주차면수', 'N버스_노선개수', '지하철역수']]
df_modeling_y = df_modeling['음주사고수']

df_scoring_x = df_scoring[['가게면적', 'CCTV수', '교차로수', '주차면수', 'N버스_노선개수', '지하철역수']]
df_scoring_y = df_scoring['음주사고수']


# train test split
train_x, test_x, train_y, test_y = train_test_split(df_modeling_x, df_modeling_y, random_state = 0)

