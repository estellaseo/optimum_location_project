#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 사고 수가 1건 이상인 grid로 각 변수별 변수중요도 확인

# =============================================================================
# DT
# =============================================================================
# [ 튜닝 전 ] 
m_dt = dt_c(random_state=0)
m_dt.fit(train_x, train_y)
m_dt.score(train_x, train_y)   # 93.44   
m_dt.score(test_x, test_y)     # 75.80 

# DT의 변수중요도 리턴 (내림차순 정렬) 
DataFrame({'변수중요도' : m_dt.feature_importances_}, index = df_modeling_x.columns).sort_values('변수중요도', ascending=False)

             변수중요도
가게면적      0.623339
CCTV수      0.181701
지하철역수     0.093988
N버스_노선개수  0.040173
교차로수      0.035746
주차면수      0.025052


# ----------------------------
# [ 튜닝 후 ]
m_dt = dt_c(max_depth=18, min_samples_split=15, splitter='random', random_state=0)
m_dt.fit(train_x, train_y)
m_dt.score(train_x, train_y)  # 82.51   
m_dt.score(test_x, test_y)    # 83.87 


# 최적화 후 DT의 변수중요도 (내림차순 정렬) 
DataFrame({'변수중요도' : m_dt.feature_importances_}, index = df_modeling_x.columns).sort_values('변수중요도', ascending=False)

            변수중요도
가게면적      0.355260
지하철역수     0.293231
교차로수      0.169650
CCTV수       0.136136
N버스_노선개수  0.045722
주차면수      0.000000



# =============================================================================
# RF  
# =============================================================================
# [ 튜닝 전 ] 
m_rf = rf_c(random_state=0)
m_rf.fit(train_x, train_y)
m_rf.score(train_x, train_y)   # 93.44
m_rf.score(test_x, test_y)     # 79.03 

# RF의 변수중요도 (내림차순 정렬)
DataFrame({'변수중요도' : m_rf.feature_importances_}, index = df_modeling_x.columns).sort_values('변수중요도', ascending=False)

            변수중요도
가게면적      0.663313
CCTV수     0.108067
지하철역수     0.083101
교차로수      0.082433
주차면수      0.045058
N버스_노선개수  0.018027

# ----------------------------
# [ 튜닝 후 ] 
m_rf = rf_c(max_depth=5, min_samples_split=2, n_estimators=10, random_state=0)
m_rf.fit(train_x, train_y)
m_rf.score(train_x, train_y)    # 84.69
m_rf.score(test_x, test_y)      # 82.25

# 최적화된  RF의 변수중요도 (내림차순 정렬) 
DataFrame({'변수중요도' : m_rf.feature_importances_}, index = df_modeling_x.columns).sort_values('변수중요도', ascending=False)


            변수중요도
가게면적      0.387480
지하철역수     0.187011
교차로수      0.178845
CCTV수     0.110161
주차면수      0.089964
N버스_노선개수  0.046540


# =============================================================================
# gbc
# =============================================================================
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# [ 튜닝 전 ] 
m_gbc = gb_c(random_state=0)
m_gbc.fit(train_x, train_y)

m_gbc.score(train_x, train_y)   # 93.44 
m_gbc.score(test_x, test_y)     # 82.25


# gbc의 변수중요도 (내림차순 정렬) 
DataFrame({'변수중요도' : m_gbc.feature_importances_}, index = df_modeling_x.columns).sort_values('변수중요도', ascending=False)

             변수중요도
가게면적      0.756643
CCTV수      0.070486
지하철역수     0.054205
교차로수      0.051820
주차면수      0.040177
N버스_노선개수  0.026669

# ----------------------------
# [ 튜닝 후 ] 
m_gbc = gb_c(learning_rate = 0.01, max_depth=3, min_samples_split=5, n_estimators=10, random_state=0)
m_gbc.fit(train_x, train_y)

m_gbc.score(train_x, train_y)  # 81.42   
m_gbc.score(test_x, test_y)    # 83.87  

# gbc의 변수중요도 (내림차순 정렬) 
DataFrame({'변수중요도' : m_gbc.feature_importances_}, index = df_modeling_x.columns).sort_values('변수중요도', ascending=False)

             변수중요도
가게면적      0.509720
교차로수      0.170660
CCTV수      0.145429
지하철역수     0.087425
주차면수      0.086767
N버스_노선개수  0.000000


# =============================================================================
# xgboost
# =============================================================================
# [ 튜닝 전 ] 
from sklearn.preprocessing import LabelEncoder
m_le = LabelEncoder()
df_modeling_y_encoded = m_le.fit_transform(df_modeling_y)

# train test split
train_x, test_x, train_y_encoded, test_y_encoded = train_test_split(df_modeling_x, df_modeling_y_encoded, random_state = 0)


m_xgb = xgb.XGBClassifier(random_state=0)
m_xgb.fit(train_x, train_y_encoded)
print(m_xgb.score(train_x, train_y_encoded))   
print(m_xgb.score(test_x, test_y_encoded))    


# 튜닝 전 xgb 변수중요도
print(DataFrame({'변수중요도' : m_xgb.feature_importances_}, index = df_modeling_x.columns).sort_values('변수중요도', ascending=False))




# ----------------------------
# [ 튜닝 후 ]
m_xgb = xgb.XGBClassifier(learning_rate=0.01, max_depth=5, n_estimators=10, random_state=0)
m_xgb.fit(train_x, train_y_encoded)
print(m_xgb.score(train_x, train_y_encoded))  
print(m_xgb.score(test_x, test_y_encoded))

# 튜닝 후 xgb 변수중요도
print(DataFrame({'변수중요도' : m_xgb.feature_importances_}, index = df_modeling_x.columns).sort_values('변수중요도', ascending=False))

             변수중요도
주차면수       0.279795
가게면적       0.214274
지하철역수     0.169488
CCTV수       0.168780
교차로수       0.167663
N버스_노선개수  0.000000




# =============================================================================
# lgbm
# =============================================================================
# 데이터 건수 많을 때 사용하는게 좋음. 
# 작은 데이터에 대해서는 과적합되기 쉬움 
# 다른 트리모델은 level-wise, lgbm은 leaf-wise (같은 층이 아니더라도 깊게 노드 만들어나감)
# https://www.kaggle.com/code/prashant111/lightgbm-classifier-in-python

# [ 튜닝 전 ]
m_lgb = lgb.LGBMClassifier(random_state=0)
m_lgb.fit(train_x, train_y)
print(m_lgb.score(train_x, train_y))  # 82.51
print(m_lgb.score(test_x, test_y))    # 85.48

# 튜닝 전 lgbm 변수중요도 (내림차순 정렬)
print(DataFrame({'변수중요도' : m_lgb.feature_importances_}, index = df_modeling_x.columns).sort_values('변수중요도', ascending=False))

           변수중요도
가게면적       1503
교차로수        619
CCTV수        568
주차면수          0
N버스_노선개수     0
지하철역수        0


# ----------------------------
# [ 튜닝 후 ]
m_lgb = lgb.LGBMClassifier(max_depth=5, learning_rate=0.001, min_data_in_leaf=3, num_boost_round=10, metric='multi_logloss', random_state=0)
m_lgb.fit(train_x, train_y)
print(m_lgb.score(train_x, train_y))   # 81.42
print(m_lgb.score(test_x, test_y))     # 83.87 

# 튜닝 후 lgbm 변수중요도 (내림차순 정렬)
print(DataFrame({'변수중요도' : m_lgb.feature_importances_}, index = df_modeling_x.columns).sort_values('변수중요도', ascending=False))

             변수중요도
가게면적         538
CCTV수         145
교차로수         67
지하철역수        48
주차면수         10
N버스_노선개수     0




# =============================================================================
# catboost
# =============================================================================
# [ 튜닝 전 ]
m_cb = CatBoostClassifier(random_seed=0)
m_cb.fit(train_x, train_y)
print(m_cb.score(train_x, train_y))  # 93.44
print(m_cb.score(test_x, test_y))    # 79.03

# 튜닝 전 cb 변수중요도
print(DataFrame({'변수중요도' : m_cb.feature_importances_}, index = df_modeling_x.columns).sort_values('변수중요도', ascending=False))
              변수중요도
가게면적       33.451728
CCTV수       29.606555
교차로수       24.273406
지하철역수      7.929942
N버스_노선개수   3.118130
주차면수        1.620238


# ----------------------------
# [ 튜닝 후 ] 
m_cb = CatBoostClassifier(depth=3, iterations=10, learning_rate=0.01, random_seed=0)
m_cb.fit(train_x, train_y)
print(m_cb.score(train_x, train_y))   # 81.42
print(m_cb.score(test_x, test_y))     # 83.87

# 튜닝 후 cb의 변수중요도 (내림차순 정렬) 
DataFrame({'변수중요도' : m_cb.feature_importances_}, index = df_modeling_x.columns).sort_values('변수중요도', ascending=False)

             변수중요도
교차로수       31.425234
지하철역수      25.158162
가게면적       18.451027
N버스_노선개수  12.097979
CCTV수        9.828240
주차면수        3.039358
 









# =============================================================================
# SVM  ->  변수중요도가 없어서 쓸모없음 
# =============================================================================
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler as minmax
from sklearn.model_selection import train_test_split


# 거리기반 모델이라 스케일링에 민감 -> 전체 컬럼 스케일링 (minmax scaling) 
m_sc = minmax()
df_modeling_x_sc = m_sc.fit_transform(df_modeling_x)

# train test split
train_x_sc, test_x_sc, train_y, test_y = train_test_split(df_modeling_x_sc, df_modeling_y, random_state=0)
     


# ----------------------------
# [ 최적화 전 ] 
m_svc = SVC()
m_svc.fit(train_x_sc, train_y)
m_svc.score(train_x_sc, train_y)   # 83.06   
m_svc.score(test_x_sc, test_y)     # 83.87


# ----------------------------
# [ 최적화 후 ] 
# 모델링 (최적 하이퍼파라미터 튜닝 완료)
m_svc = SVC(C=0.001, gamma=0.001)
m_svc.fit(train_x_sc, train_y)
m_svc.score(train_x_sc, train_y)   # 81.42
m_svc.score(test_x_sc, test_y)     # 83.87



