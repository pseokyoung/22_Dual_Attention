import random

import numpy as np
import random

def data2series(data, history_length, history_var, future_length, future_var,
                step=1, start_idx=0, end_idx=None):
    
    history_data = data[history_var].values
    future_data = data[future_var].values
   
    history_series = []
    future_series = []
    
    start_idx = start_idx + history_length
    if end_idx is None:
        end_idx = len(data) - future_length
    else:
        end_idx = end_idx - future_length
        
    for i in range(start_idx, end_idx):
        history_series.append(history_data[range(i-history_length, i, step)])
        if future_length == 1:
            future_series.append(future_data[i])
        else:
            future_series.append(future_data[i : i+future_length : step])
    
    return np.array(history_series), np.array(future_series)

def ctsfilter(df, min_length = 50):
    cts_ranges = []

    for i in range(len(df)):
        if not i:
            start_idx = df.index[i]
            temp_idx = start_idx
            temp_idx += 1
        elif temp_idx == df.index[i]:
            temp_idx += 1
        else:
            end_idx = df.index[i-1]
            if end_idx - start_idx > min_length:
                cts_ranges.append(range(start_idx, end_idx))
            start_idx = df.index[i]
            temp_idx = start_idx
            temp_idx += 1
    print(f"total number of continuous range: {len(cts_ranges)}")
    
    cts_df_list = []
    for cts_range in cts_ranges:
        cts_df_list.append(df.loc[cts_range])
    return cts_df_list