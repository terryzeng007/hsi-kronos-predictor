# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 09:34:02 2026

@author: Administrator

v1
    -读取kronos文件夹预测价格进行回测
    -恒指
    
"""

import pandas as pd
import numpy as np
import sys, os
import matplotlib.pyplot as plt

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

project_path = 'D:/owncloud_117/1_Import_Code'
sys.path.append(project_path)
os.chdir(project_path)
import Terry as tz
import Strategy_Function as sf
import TA_Signal_Support_v3 as tss


project_path = 'D:/Git_Project/agent2_graph/predict_rolling/'
sys.path.append(project_path)
os.chdir(project_path)

#%%




#%%
# para
target = 'half_year'
buy_p = 0.15
sell_p = -0.15
st = '2011-05-05'
end = '2026-01-12'

# data clean
df_folder = tz.get_all_files(target)
df_all = None

for ex in df_folder:
    df_i = pd.read_csv(target + '/' + ex)
    
    judge = df_i['pred_close'].iloc[-1]/df_i['pred_close'].iloc[0]   -1
    if judge>buy_p:
        df_i['judge'] = 1
    elif judge<sell_p:
        df_i['judge'] = -1
    else:
        df_i['judge'] = 0
    
    df_all = pd.concat([df_all, df_i], axis=0)

# graph
df_all.index = pd.to_datetime(df_all['date'])
df_all = df_all.loc[st:end]
df_all[['close','pred_close']].plot(grid=True, title=target)


# teturn
df_all['r'] = df_all['close']/df_all['close'].shift(1)-1
df_all['r_hsi'] = (1+df_all['r']).cumprod()

df_all['r_long'] = np.where(df_all['judge']<=0, 0, 1)
df_all['r_long'] = df_all['r']*df_all['r_long']
df_all['r_long'] = (1+df_all['r_long']).cumprod()

df_all['r_short'] = np.where(df_all['judge']<0, -1, 0)
df_all['r_short'] = df_all['r']*df_all['r_short']
df_all['r_short'] = (1+df_all['r_short']).cumprod()
 
df_all['r_both'] = df_all['r']*df_all['judge']
df_all['r_both'] = (1+df_all['r_both']).cumprod()

df_all[['r_hsi','r_both','r_long','r_short']].plot(grid=True, title=target)

df_all[['judge']].plot(grid=True, title=target + '-position')

df_all['judge'].value_counts()


#%%