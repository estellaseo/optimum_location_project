#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# =============================================================================
# dt
# =============================================================================
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth': range(2,30),
              'min_samples_split': range(2,30),
              'splitter': ['best', 'random']}

grid_dt = GridSearchCV(m_dt, param_grid = parameters, cv = 5)

grid_dt.fit(train_x, train_y)

result = DataFrame(grid_dt.cv_results_['params'])
result['mean_test_score'] = grid_dt.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score', ascending=False).head(20) 


# 아래는 결과값 
       max_depth     min_samples_split  splitter  mean_test_score
923          18                 15   random         0.814264
327           7                 25   random         0.808859
1307         25                 11   random         0.808859
127           4                  9   random         0.808859
1325         25                 20   random         0.808859
713          14                 22   random         0.808859
615          12                 29   random         0.808859
723          14                 27   random         0.808859
357           8                 12   random         0.808859
1341         25                 28   random         0.808859
53            2                 28   random         0.808859
223           5                 29   random         0.808859
50            2                 27     best         0.808859
46            2                 25     best         0.808859
977          19                 14   random         0.808859
42            2                 23     best         0.808859
1367         26                 13   random         0.808859
805          16                 12   random         0.808859
705          14                 18   random         0.808859
987          19                 19   random         0.808859



# =============================================================================
# rf
# =============================================================================
parameters = {'max_depth': range(2,30),
              'min_samples_split': range(2,30),
              'n_estimators' : range(10,100,10)}

grid_rf = GridSearchCV(m_rf, param_grid = parameters, cv = 5)

grid_rf.fit(train_x, train_y)

result = pd.DataFrame(grid_rf.cv_results_['params'])
result['mean_test_score'] = grid_rf.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score', ascending=False).head(20)


# 아래는 결과값 
          max_depth  min_samples_split  n_estimators  mean_test_score
756           5                  2            10         0.814414
0             2                  2            10         0.814264
3907         17                 16            20         0.814264
3914         17                 16            90         0.814264
3913         17                 16            80         0.814264
3912         17                 16            70         0.814264
3911         17                 16            60         0.814264
3910         17                 16            50         0.814264
3908         17                 16            30         0.814264
3905         17                 15            90         0.814264
3919         17                 17            50         0.814264
3904         17                 15            80         0.814264
3903         17                 15            70         0.814264
3901         17                 15            50         0.814264
3900         17                 15            40         0.814264
3899         17                 15            30         0.814264
3898         17                 15            20         0.814264
3896         17                 14            90         0.814264
3917         17                 17            30         0.814264
3920         17                 17            60         0.814264



# =============================================================================
# gbc
# =============================================================================
# 시도1)
parameters = {'max_depth': range(3,20),
              'min_samples_split' : range(5,10),
              'n_estimators' : range(10,15),
              'learning_rate' : [0.05, 0.1]}

grid_gbc = GridSearchCV(m_gbc, param_grid = parameters, cv = 2)

grid_gbc.fit(train_x, train_y)

result = pd.DataFrame(grid_gbc.cv_results_['params'])
result['mean_test_score'] = grid_gbc.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score', ascending=False).head(20)


# 아래는 결과값 
       learning_rate  max_depth  min_samples_split  n_estimators  mean_test_score
16           0.05          3                  8            11         0.803333
10           0.05          3                  7            10         0.803333
22           0.05          3                  9            12         0.803333
21           0.05          3                  9            11         0.803333
20           0.05          3                  9            10         0.803333
15           0.05          3                  8            10         0.803333
11           0.05          3                  7            11         0.803333
23           0.05          3                  9            13         0.792344
17           0.05          3                  8            12         0.792344
12           0.05          3                  7            12         0.792344
0            0.05          3                  5            10         0.792344
6            0.05          3                  6            11         0.792344
5            0.05          3                  6            10         0.792344
42           0.05          4                  8            12         0.786849
47           0.05          4                  9            12         0.786849
65           0.05          5                  8            10         0.786849
70           0.05          5                  9            10         0.786849
71           0.05          5                  9            11         0.786849
35           0.05          4                  7            10         0.786849
36           0.05          4                  7            11         0.786849



# 시도2) parameters 키 밸류 변경해서 다시 적용 
parameters = {'max_depth': range(3,20),
              'min_samples_split' : range(5,10),
              'n_estimators' : range(10,15),
              'learning_rate' : [0.05, 0.01]}

grid_gbc = GridSearchCV(m_gbc, param_grid = parameters, cv = 2)

grid_gbc.fit(train_x, train_y)

result = pd.DataFrame(grid_gbc.cv_results_['params'])
result['mean_test_score'] = grid_gbc.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score', ascending=False).head(20)


# 아래는 결과값 
       learning_rate  max_depth  min_samples_split  n_estimators  mean_test_score
425           0.01          3                  5            10         0.814202
559           0.01          8                  6            14         0.814202
561           0.01          8                  7            11         0.814202
562           0.01          8                  7            12         0.814202
563           0.01          8                  7            13         0.814202
564           0.01          8                  7            14         0.814202
565           0.01          8                  8            10         0.814202
566           0.01          8                  8            11         0.814202
567           0.01          8                  8            12         0.814202
568           0.01          8                  8            13         0.814202
569           0.01          8                  8            14         0.814202
570           0.01          8                  9            10         0.814202
571           0.01          8                  9            11         0.814202
572           0.01          8                  9            12         0.814202
573           0.01          8                  9            13         0.814202
574           0.01          8                  9            14         0.814202
575           0.01          9                  5            10         0.814202
576           0.01          9                  5            11         0.814202
577           0.01          9                  5            12         0.814202
578           0.01          9                  5            13         0.814202

# =============================================================================
# xgboost
# =============================================================================
# 아래 파라미터 말고 다른 범위, 다른 하이퍼파라미터도 추가해서 해봄. 그래도 점수가 비슷 
parameters = {'max_depth':[5,7,10,15],
              'learning_rate' : [0.01, 0.03, 0.05],
              'n_estimators' : [10,30,50,70,100],
              }


grid_xgb = GridSearchCV(estimator=m_xgb, param_grid = parameters, cv = 2, n_jobs=-1)
grid_xgb.fit(train_x, train_y)
result = DataFrame(grid_xgb.cv_results_['params'])
result['mean_test_score'] = grid_xgb.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score', ascending=False).head(20)


# 결과
learning_rate	max_depth	n_estimators	mean_test_score
     0.01	        5	         10	       0.770724
     0.01	       15	         50	       0.770724
     0.05	       15	         10	       0.770724
     0.05	       10	         10	       0.770724
     0.05	        7	         10	       0.770724
     0.05	        5	         10	       0.770724
     0.03	       15	         30	       0.770724
     0.03	       15	         10	       0.770724
     0.03	       10	         30	       0.770724



# =============================================================================
# lgbm
# =============================================================================

parameters = {'max_depth':[5,7,10,13,15], 
              'min_data_in_leaf':[3,5,7,10],
              'learning_rate':[0.001,0.005,0.01,0.05],
              'num_boost_round':[10,50,70,100],
              'metric':['mse','multi_logloss']}

grid_lgb = GridSearchCV(estimator=m_lgb, param_grid = parameters, cv = 2, n_jobs=-1)
grid_lgb.fit(train_x, train_y)
result = DataFrame(grid_lgb.cv_results_['params'])
result['mean_test_score'] = grid_lgb.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score', ascending=False).head(20)

# 결과
learning_rate	max_depth	 metric	   min_data_in_leaf	num_boost_round	mean_test_score
    0.001	        5	   multi_logloss	     3	            10	       0.814202
    0.010	       	5	   multi_logloss	     5	            70	       0.814202
    0.010	       	5	   multi_logloss	     7	            10	       0.814202
    0.010	       	5	   multi_logloss	     7	            50	       0.814202
    0.010	        5	   multi_logloss	     7	            70	       0.814202
    0.010	        5  	 multi_logloss	     7	           100	       0.814202
    0.010	        5	   multi_logloss	    10	            10	       0.814202
    0.010	        5	   multi_logloss	    10	            50	       0.814202
    0.010	        5	   multi_logloss	    10	            70	       0.814202
    0.010	        5	   multi_logloss	    10	           100	       0.814202
    0.010	        7	   multi_logloss	     3	            10	       0.814202
    0.010	        7	   multi_logloss	     3	            50	       0.814202
    0.010	        7	   multi_logloss	     5	            10	       0.814202
    0.010	        7	   multi_logloss	     5	            50	       0.814202
    0.010	        7	   multi_logloss	     5	            70	       0.814202
    0.010	        7	   multi_logloss	     5	           100	       0.814202
    0.010	        7	   multi_logloss	     7	            10	       0.814202
    0.010	        7	   multi_logloss	     7	            50	       0.814202
    0.010	        7	   multi_logloss	     7	            70	       0.814202
    0.010	        7	   multi_logloss	     7	           100	       0.814202





# =============================================================================
# catboost
# =============================================================================
parameters = {'depth':[3,5,10,15], 
              'learning_rate':[0.01,0.03,0.05],
              'iterations':[10,50,100,300]}

grid_cb = GridSearchCV(estimator=m_cb, param_grid = parameters, cv = 2, n_jobs=-1)
grid_cb.fit(train_x, train_y)
result = DataFrame(grid_cb.cv_results_['params'])
result['mean_test_score'] = grid_cb.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score', ascending=False).head(20)

# 결과 
depth	iterations	learning_rate	mean_test_score
 3	      10	      0.01	     0.8142021022455805
10	     100      	0.05	     0.8142021022455805
 3	      10	      0.03       0.8142021022455805
10	      10	      0.03	     0.8142021022455805
10	      10	      0.05	     0.8142021022455805
10      	50	      0.01	     0.8142021022455805
10	      50	      0.03	     0.8142021022455805
10	      50	      0.05	     0.8142021022455805
10	     100	      0.01	     0.8142021022455805
15	      10	      0.01	     0.8142021022455805
5	       100	      0.03	     0.8142021022455805
15	      10	      0.03	     0.8142021022455805
15	      10	      0.05	     0.8142021022455805
15	      50	      0.01	     0.8142021022455805
15	      50	      0.03	     0.8142021022455805
15	      50	      0.05	     0.8142021022455805
15	     100	      0.01	     0.8142021022455805
15	     100	      0.03	     0.8142021022455805
5	       300	      0.01	     0.8142021022455805
10	      10	      0.01	     0.8142021022455805






# =============================================================================
# SVM
# =============================================================================

parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
             'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_svc = GridSearchCV(m_svc, param_grid= parameters, cv=5)
grid_svc.fit(train_x_sc, train_y)

result = pd.DataFrame(grid_svc.cv_results_['params'])
result['mean_test_score'] = grid_svc.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score', ascending=False).head(20)


# 아래는 결과값 
#           C    gamma  mean_test_score
# 0     0.001    0.001         0.814264
# 16    0.100   10.000         0.814264
# 32  100.000    0.100         0.814264
# 31  100.000    0.010         0.814264
# 30  100.000    0.001         0.814264
# 26   10.000    0.100         0.814264
# 25   10.000    0.010         0.814264
# 24   10.000    0.001         0.814264
# 23    1.000  100.000         0.814264
# 22    1.000   10.000         0.814264
# 21    1.000    1.000         0.814264
# 20    1.000    0.100         0.814264
# 19    1.000    0.010         0.814264
# 1     0.001    0.010         0.814264
# 17    0.100  100.000         0.814264
# 18    1.000    0.001         0.814264
# 15    0.100    1.000         0.814264
# 7     0.010    0.010         0.814264
# 2     0.001    0.100         0.814264
# 3     0.001    1.000         0.814264



