import numpy as np
import math

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

class RNN():
    def __init__(self,
                 history_arr, future_arr):
        self.history_arr = history_arr
        self.future_arr = future_arr
        
    def train_test_split(self, test_size, test_num):
        # test_size: 테스트 데이터의 비율
        # test_num: 테스트 시작 위치 지정
        
        num_sample = self.history_arr.shape[0]
        num_test = int(num_sample*test_size)
        
        start_idx   = num_test*test_num
        end_idx     = num_test*(test_num+1)
        
        assert end_idx < num_sample 
        
        self.history_train  = np.concatenate([self.history_arr[range(0       ,start_idx),:,:],
                                              self.history_arr[range(end_idx ,num_sample),:,:],
                                              axis=0]
        self.history_test   = self.history_arr[range(start_idx,end_idx),:,:]
        
        self.future_train   = np.concatenate([self.future_arr[range(0       ,start_idx),:,:],
                                              self.future_arr[range(end_idx ,num_sample),:,:],
                                              axis=0]
        self.future_test    = self.future_arr[range(start_idx,end_idx),:,:]
        
    def scaling(self, scaler_type='standard'):
        if scaler_type == 'standard':
            self.history_mean = self.history_train[:,0,:].mean(axis=0)
            self.history_std  = self.history_train[:,0,:].std(axis=0)
            self.future_mean  = self.future_train[:,0,:].mean(axis=0)
            self.future_std   = self.future_train[:,0,:].std(axis=0)
            
            self.history_train_sc = (self.history_train-self.history_mean)/self.history_std
            self.history_test_sc  = (self.history_test-self.history_mean)/self.history_std
            self.future_train_sc  = (self.future_train-self.future_mean)/self.future_std
            self.future_test_sc   = (self.future_test-self.future_mean)/self.future_std
            
        elif scaler_type == 'minmax':
            self.history_min = self.history_train[:,0,:].min(axis=0)
            self.history_max = self.history_train[:,0,:].max(axis=0)
            self.future_min  = self.future_train[:,0,:].min(axis=0)
            self.future_max  = self.future_train[:,0,:].max(axis=0)
            
            self.history_train_sc = (self.history_train-self.history_min)/(self.history_max-self.history_min)
            self.history_test_sc  = (self.history_test-self.history_min)/(self.history_max-self.history_min)
            self.future_train_sc  = (self.future_train-self.future_min)/(self.future_max-self.future_min)
            self.future_test_sc   = (self.future_test-self.future_min)/(self.future_max-self.future_min)
            
    def model(self, history_shape, future_shape, num_layers, num_neurons, rnn_type, activation='relu'):
        if rnn_type == 's2s_gru':
            input = Input(shape=(x_input.shape[1], x_input.shape[2]))
            output = Input(shape=(y_output.shape[1], y_output.shape[2]))

            if num_layers == 1:
                hidden_state_en = input

            else:
                #인코더 은닉층
                for layer in range(num_layers-1):
                    if layer == 0:
                        hidden_state_en = GRU(num_neurons, 
                                            return_sequences=True,
                                            return_state=False)(input)
                    else:
                        hidden_state_en = GRU(num_neurons, 
                                            return_sequences=True,
                                            return_state=False)(hidden_state_en)

            encoder_last_h1, encoder_last_c = GRU(num_neurons, 
                                                                    return_sequences=False,
                                                                    return_state=True)(hidden_state_en)

            decoder_input = RepeatVector(output.shape[1])(encoder_last_h1)

            if num_layers == 1:
                hidden_state_de = decoder_input

            else:
                #디코더 은닉층
                for layer in range(num_layers-1):
                    if layer == 0:
                        hidden_state_de = GRU(num_neurons, 
                                            return_sequences=True,
                                            return_state=False)(decoder_input)
                    else:
                        hidden_state_de = GRU(num_neurons, 
                                            return_sequences=True,
                                            return_state=False)(hidden_state_de)

            decoder = GRU(num_neurons,
                        return_sequences = True,
                        return_state=False)(hidden_state_de)  #, initial_state=[encoder_last_h1,encoder_last_c ])

            out = TimeDistributed(Dense(output.shape[2]))(decoder)

            if dnn_neurons:
                out = Dense(Len_Futr)(out)
                out = Dense(output.shape[2])(out)

            model = Model(inputs=input, outputs=out)

            optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
            model.compile(loss='mse',
                        optimizer = optimizer)

            return model