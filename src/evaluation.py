import numpy as np

def r_squared(prediction, actual):    
    actual_mean = np.mean(actual, axis=0)
    SSR = np.sum( (prediction - actual)**2 , axis=0)    
    TSS = np.sum( (actual - actual_mean )**2, axis=0 )    
    
    R_square = 1 - SSR/TSS
    return R_square
                          
def rmse(prediction, actual):
    rmse = np.sqrt(np.mean((prediction - actual)**2, axis=0))
    return rmse

def nrmse(prediction, actual):
    rmse = np.sqrt(np.mean((prediction - actual)**2, axis=0))
    min_hat = np.min(actual, axis=0)
    max_hat = np.max(actual, axis=0)
    nrmse = rmse/(max_hat - min_hat)*100
    return nrmse