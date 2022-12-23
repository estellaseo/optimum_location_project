#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 변수 스케일링 (minmax로 진행 - 음수값이 없도록) 
from sklearn.preprocessing import MinMaxScaler as minmax
m_sc = minmax()
df_scoring_x_sc = m_sc.fit_transform(df_scoring_x)

  # df_scoring은 음주사고수 == 0 인 그리드 



# =============================================================================
# 스케일링한 변수에 변수중요도를 곱해서 위험도 스코어 계산  
# =============================================================================

df2 = df_scoring[:]   

# DT 변수중요도로 만든 위험도 스코어 
df2['index_dt'] = m_dt.feature_importances_[0]*df_scoring_x_sc[:,0] - m_dt.feature_importances_[1]*df_scoring_x_sc[:,1] + m_dt.feature_importances_[2]*df_scoring_x_sc[:,2] + m_dt.feature_importances_[3]*df_scoring_x_sc[:,3] - m_dt.feature_importances_[4]*df_scoring_x_sc[:,4] + m_dt.feature_importances_[5]*df_scoring_x_sc[:,5]

# RF 변수중요도로 만든 위험도 스코어 
df2['index_rf'] = m_rf.feature_importances_[0]*df_scoring_x_sc[:,0] - m_rf.feature_importances_[1]*df_scoring_x_sc[:,1] + m_rf.feature_importances_[2]*df_scoring_x_sc[:,2] + m_rf.feature_importances_[3]*df_scoring_x_sc[:,3] - m_rf.feature_importances_[4]*df_scoring_x_sc[:,4] + m_rf.feature_importances_[5]*df_scoring_x_sc[:,5]

# gbm 변수중요도로 만든 위험도 스코어  
df2['index_gbm'] = m_gbc.feature_importances_[0]*df_scoring_x_sc[:,0] - m_gbc.feature_importances_[1]*df_scoring_x_sc[:,1] + m_gbc.feature_importances_[2]*df_scoring_x_sc[:,2] + m_gbc.feature_importances_[3]*df_scoring_x_sc[:,3] - m_gbc.feature_importances_[4]*df_scoring_x_sc[:,4] + m_gbc.feature_importances_[5]*df_scoring_x_sc[:,5]



# =============================================================================
# 각 모델로 y_pred 값 뽑아서 df2 데이터에 추가  
# =============================================================================

df2['y_pred_dt'] = m_dt.predict(df_scoring_x)
df2['y_pred_rf'] = m_rf.predict(df_scoring_x)
df2['y_pred_gbm'] = m_gbc.predict(df_scoring_x)


# df2 데이터를 index_221108.csv 파일로 저장 
df2.to_csv('index_221109.csv', encoding='cp949')