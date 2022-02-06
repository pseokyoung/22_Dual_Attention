import numpy as np
import math
import time

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from keras.callbacks import EarlyStopping

from src import models


def super_attention(delta, future_size, future_num, att_type):
    start_factor = 1 + delta
    end_factor = 1
    num_points = future_size    
    
    if att_type == 'linear':
        factor = np.linspace(start_factor, end_factor, num_points)
        factor = np.array([factor, factor, factor]).T
        factor = factor[np.newaxis,:,:]

    elif att_type == 'exp': 
        factor = np.array([(start_factor)**((num_points-k-1)/(num_points-1)) for k in range(num_points)])
        factor = np.array([factor, factor, factor]).T
        factor = factor[np.newaxis,:,:]

    else:
        print("invalid delta")
        
    return factor

class RNN():
    def __init__(self,
                 history_arr, future_arr):
        self.history_arr = history_arr
        self.future_arr = future_arr
        
        self.history_train = []
        self.history_valid = []
        self.history_test  = []
        self.future_train  = []
        self.future_valid  = []
        self.future_test   = []
        
        self.history_train_sc = []
        self.history_valid_sc = []
        self.history_test_sc  = []
        self.future_train_sc  = []
        self.future_valid_sc  = []
        self.future_test_sc   = []
        
    def train_test(self, test_size, test_num):
        # test_size: 테스트 데이터의 비율
        # test_num: 테스트 시작 위치 지정
        num_sample = self.history_arr.shape[0]  # 전체 샘플 수
        num_test = int(num_sample*test_size)    # 테스트 샘플 수
        
        if test_num == -1:
            start_idx   = num_sample - num_test
            end_idx     = num_sample            
        else:
            start_idx   = num_test*test_num
            end_idx     = num_test*(test_num+1)
            
        assert end_idx <= num_sample 
        
        self.history_train  = np.concatenate([self.history_arr[range(0       ,start_idx),:,:],
                                                self.history_arr[range(end_idx ,num_sample),:,:]], axis=0)
        self.history_test   = self.history_arr[range(start_idx,end_idx),:,:]
        
        self.future_train   = np.concatenate([self.future_arr[range(0       ,start_idx),:,:],
                                                self.future_arr[range(end_idx ,num_sample),:,:]], axis=0)
        self.future_test    = self.future_arr[range(start_idx,end_idx),:,:]
        
    def train_valid(self, valid_size, random_state=42):
        self.history_train, self.history_valid = train_test_split(self.history_train, test_size=valid_size, random_state=random_state, shuffle=True)
        self.future_train, self.future_valid = train_test_split(self.future_train, test_size=valid_size, random_state=random_state, shuffle=True)        

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
            
            if len(self.history_valid):
                self.history_valid_sc  = (self.history_valid-self.history_mean)/self.history_std                
                self.future_valid_sc   = (self.future_valid-self.future_mean)/self.future_std
                            
        elif scaler_type == 'minmax':
            self.history_min = self.history_train[:,0,:].min(axis=0)
            self.history_max = self.history_train[:,0,:].max(axis=0)
            self.future_min  = self.future_train[:,0,:].min(axis=0)
            self.future_max  = self.future_train[:,0,:].max(axis=0)
            
            self.history_train_sc = (self.history_train-self.history_min)/(self.history_max-self.history_min)
            self.history_test_sc  = (self.history_test-self.history_min)/(self.history_max-self.history_min)
            self.future_train_sc  = (self.future_train-self.future_min)/(self.future_max-self.future_min)
            self.future_test_sc   = (self.future_test-self.future_min)/(self.future_max-self.future_min)
            
            if len(self.history_valid):
                self.history_valid_sc  = (self.history_valid-self.history_min)/(self.history_max-self.history_min)          
                self.future_valid_sc   = (self.future_valid-self.future_min)/(self.future_max-self.future_min)          
                
    def unscaling(self, history_arr_sc, future_arr_sc, scaler_type='standard'):
        if scaler_type == 'standard':
            history_arr = history_arr_sc*self.history_std + self.history_mean
            future_arr  = future_arr_sc*self.future_std + self.future_mean
                            
        elif scaler_type == 'minmax':
            history_arr = history_arr_sc*(self.history_max-self.history_min) + self.history_min
            future_arr  = future_arr_sc*(self.future_max-self.future_min) + self.future_min 
                
        return history_arr, future_arr
           
    def build_model(self, num_layers, num_neurons, dense_layers, dense_neurons, model_type, factor=[], activation='relu'):
        # https://github.com/KerasKorea/KEKOxTutorial/blob/master/28_A_ten-minute_introduction_to_sequence-to-sequence_learning_in_Keras.md
        history_size = self.history_train.shape[1]
        history_num = self.history_train.shape[2]
        future_size  = self.future_train.shape[1]
        future_num  = self.future_train.shape[2]        
        
        if model_type == 'seq2seq_lstm':
            self.model = models.seq2seqLSTM(history_size, history_num, future_size, future_num, num_layers, num_neurons, dense_layers, dense_neurons)
        elif model_type == 'seq2seq_gru':
            self.model = models.seq2seqGRU(history_size, history_num, future_size, future_num, num_layers, num_neurons, dense_layers, dense_neurons)
        elif model_type == 'att_seq2seq_lstm':
            self.model = models.ATTseq2seqLSTM(history_size, history_num, future_size, future_num, num_layers, num_neurons, dense_layers, dense_neurons)
        elif model_type == 'att_seq2seq_gru':
            self.model = models.ATTseq2seqGRU(history_size, history_num, future_size, future_num, num_layers, num_neurons, dense_layers, dense_neurons)  
        elif model_type == 'datt_seq2seq_lstm':
            self.model = models.DATTseq2seqLSTM(history_size, history_num, future_size, future_num, factor, num_layers, num_neurons, dense_layers, dense_neurons)      
        elif model_type == 'datt_seq2seq_gru':
            self.model = models.DATTseq2seqGRU(history_size, history_num, future_size, future_num, factor, num_layers, num_neurons, dense_layers, dense_neurons)     
            
    def train(self, epochs=10000, verbose=2, batch_size=32, patience=10, monitor='val_loss'):
        early_stopping_cb = EarlyStopping(patience=patience, restore_best_weights= True, monitor=monitor)
        
        time_start = time.time()
        self.history = self.model.fit(self.history_train_sc, self.future_train_sc, 
                                      epochs=epochs, callbacks=[early_stopping_cb], verbose=verbose, batch_size=batch_size, 
                                      validation_data=(self.history_valid_sc, self.future_valid_sc))
        time_end = time.time()
        self.train_time = time_start - time_end
        
    def save_model(self, save_path):
        self.model.save(save_path)