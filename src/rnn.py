import numpy as np

def super_attention(delta, pred_size, att_type):
    start_factor = 1 + delta
    end_factor = 1
    num_points = pred_size    
    
    if att_type == 'linear':
        factor = np.linspace(start_factor, end_factor, num_points)[np.newaxis,:,np.newaxis]

    elif att_type == 'exp': 
        factor = np.array([(start_factor)**((num_points-k-1)/(num_points-1)) 
                           for k in range(num_points)])[np.newaxis,:,np.newaxis]

    else:
        print("invalid delta")
        
    return factor